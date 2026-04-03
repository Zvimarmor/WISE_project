import json
import os
import sys
import random
import torch
import numpy as np
from tqdm import tqdm

# Add EasyEdit to path
sys.path.append('EasyEdit')
from easyeditor import BaseEditor, WISEHyperParams

# Optional imports for better metrics
try:
    from sentence_transformers import SentenceTransformer, util
    S_TRANSFORMERS_AVAILABLE = True
except ImportError:
    S_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence_transformers not available. Semantic similarity will be disabled.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not available. ROUGE metrics will be disabled.")

def verify_wise_llama():
    # 1. Config and Setup (Default to Llama-3.1-Instruct)
    hparams_path = 'EasyEdit/hparams/WISE/llama-3.1-8b-instruct.yaml'
    print(f"Loading config from {hparams_path}")
    if not os.path.exists(hparams_path):
        print(f"Error: Config not found at {hparams_path}")
        return

    hparams = WISEHyperParams.from_hparams(hparams_path)
    hparams.sequential_edit = True
    hparams.sticky_routing = True
    hparams.use_chat_template = True  # CRITICAL for Instruct models
    
    # 2. Data Loading
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/hallucination/wikibio-test-all.json', help='Path to the dataset (json)')
    parser.add_argument('--output_name', type=str, default='llama_validation_results')
    parser.add_argument('--max_samples', type=int, default=40, help='Maximum number of samples to evaluate.')
    parser.add_argument('--add_eos', action='store_true', help='Append EOS string (if model doesnt stop naturally).')
    args, unknown = parser.parse_known_args()

    data_path = args.data_path
    print(f"Loading data from {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Selection
    random.seed(42)
    random.shuffle(data)
    num_samples = min(args.max_samples, len(data))
    samples = data[:num_samples]
    print(f"Selected {num_samples} samples for Llama validation.")

    def get_val(r, keys):
        for k in keys:
            if k in r: return r[k]
        return None

    prompts = [str(get_val(r, ['text', 'prompt'])) for r in samples]
    targets = []
    for r in samples:
        t = get_val(r, ['labels', 'target_new'])
        target_str = str(t[0]) if isinstance(t, list) and len(t) > 0 else str(t)
        # Check for Llama 2 (</s>), GPT-J (<|endoftext|>), and Llama 3 (<|eot_id|>)
        if args.add_eos:
            if not any(stop in target_str for stop in ["</s>", "<|endoftext|>", "<|eot_id|>"]):
                target_str += " <|eot_id|>"
        targets.append(target_str)
            
    subjects = [str(get_val(r, ['concept', 'subject'])) for r in samples]
    def get_loc_prompt(r):
        loc = get_val(r, ['locality', 'locality_prompt'])
        if isinstance(loc, str): return loc
        return 'Who is the president of the US?'

    loc_prompts = [get_loc_prompt(r) for r in samples]

    # 3. Patch for Generation (Ensuring Chat Template is used)
    from easyeditor.evaluate import evaluate_utils
    import easyeditor.evaluate as eval_pkg
    
    def patched_test_generation_quality(model, tok, prefixes, max_out_len, vanilla_generation=False):
        # Apply Llama Chat Template to prefixes
        chat_prefixes = []
        for p in prefixes:
            chat = [{"role": "user", "content": p}]
            # apply_chat_template returns a string if tokenize=False and add_generation_prompt=True
            templated = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            chat_prefixes.append(templated)
        
        gen_texts = evaluate_utils.generate_fast(model, tok, chat_prefixes, n_gen_per_prompt=1, max_out_len=300, vanilla_generation=True)
        return {"ngram_entropy": 0.0, "generated_text": gen_texts}
        
    evaluate_utils.test_generation_quality = patched_test_generation_quality

    # 4. Initialize Editor
    # Force float16 to save memory (especially for Llama-3.1-8B)
    editor = BaseEditor.from_hparams(hparams)
    editor.model.to(torch.float16)  # Ensure everything is in fp16

    # 5. Model Editing
    print(f"\n========== Starting {num_samples}-Story Llama-Chat Run ==========\n")
    metrics, _, _, _ = editor.edit(
        prompts=prompts,
        target_new=targets,
        subject=subjects,
        loc_prompts=loc_prompts,
        ground_truth=['</s>'] * len(prompts),
        sequential_edit=True,
        test_generation=True,
        verbose=True  # Helpful for first instruct run
    )

    # 6. Optimized Metric Calculation (Batch SBERT)
    print("\nCalculating metrics...")
    if S_TRANSFORMERS_AVAILABLE:
        st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    all_gen_texts = []
    all_targets = []
    for i, m in enumerate(metrics):
        prompt = m['requested_rewrite']['prompt']
        # Llama's generate_fast already includes the prompt usually, so we strip it.
        # But for Chat Template, it might be inside [INST].
        post = m.get('post', {})
        fluency = post.get('fluency', {})
        gen_text = fluency.get('generated_text', [""])[0]
        
        # Cleanup
        if "[/INST]" in gen_text:
            gen_text = gen_text.split("[/INST]")[-1].strip()
        elif prompt in gen_text:
            gen_text = gen_text[len(prompt):].strip()
        else:
            gen_text = gen_text.strip()
        
        all_gen_texts.append(gen_text)
        all_targets.append(targets[i])

    similarities = [0.0] * len(all_gen_texts)
    if S_TRANSFORMERS_AVAILABLE:
        print(f"Encoding {len(all_gen_texts)} pairs...")
        gen_embs = st_model.encode(all_gen_texts, convert_to_tensor=True)
        target_embs = st_model.encode(all_targets, convert_to_tensor=True)
        similarities = torch.diagonal(util.cos_sim(gen_embs, target_embs)).tolist()

    results = []
    for i in range(len(metrics)):
        row = {
            "step": i,
            "subject": subjects[i],
            "prompt": prompts[i],
            "target": targets[i],
            "generated_text": all_gen_texts[i],
            "semantic_similarity": similarities[i],
            "rougeL": float(scorer.score(targets[i], all_gen_texts[i])['rougeL'].fmeasure) if ROUGE_AVAILABLE else 0.0
        }
        results.append(row)

    # 7. Save
    res_dir = 'results/llama_instruct_test'
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, f"{args.output_name}.json"), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Done! Results: {res_dir}/{args.output_name}.json")

if __name__ == "__main__":
    verify_wise_llama()

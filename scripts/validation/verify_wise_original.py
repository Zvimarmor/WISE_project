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
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not available. ROUGE metrics will be disabled.")

def verify_wise_original():
    # 1. Config and Setup
    hparams_path = 'EasyEdit/hparams/WISE/gpt-j-6B.yaml'
    print(f"Loading config from {hparams_path}")
    if not os.path.exists(hparams_path):
        print(f"Error: Config not found at {hparams_path}")
        return

    hparams = WISEHyperParams.from_hparams(hparams_path)
    hparams.sequential_edit = True
    hparams.act_ratio = 0.8  # Robust ratio for GPT-J
    hparams.sticky_routing = True
    
    # 2. Data Loading
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/hallucination/wikibio-test-all.json', help='Path to the dataset (json)')
    parser.add_argument('--results_folder', type=str, default='results/default_run', help='Folder to save results.')
    parser.add_argument('--output_name', type=str, default='results', help='Filename for the JSON output.')
    parser.add_argument('--num_samples', type=int, default=200, help='Maximum number of samples to evaluate.')
    parser.add_argument('--add_eos', action='store_true', help='Append <|endoftext|> to target strings.')
    args, unknown = parser.parse_known_args()

    data_path = args.data_path
    print(f"Loading data from {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Selection logic
    random.seed(42)
    random.shuffle(data)
    num_samples = min(args.num_samples, len(data))
    samples = data[:num_samples] if 'temporal' in data_path.lower() or 'extrap' in data_path.lower() else random.sample(data, num_samples)
    print(f"Selected {num_samples} samples for validation.")

    def get_val(r, keys):
        for k in keys:
            if k in r: return r[k]
        return None

    prompts = [str(get_val(r, ['text', 'prompt'])) for r in samples]
    
    # Targets are clean strings — no EOS appended at training time.
    targets = []
    for r in samples:
        t = get_val(r, ['labels', 'target_new'])
        if isinstance(t, list) and len(t) > 0:
            target_str = str(t[0])
        else:
            target_str = str(t)
        
        if args.add_eos and not target_str.endswith("<|endoftext|>"):
            target_str += " <|endoftext|>"
        targets.append(target_str)
            
    subjects = [str(get_val(r, ['concept', 'subject'])) for r in samples]
    def get_loc_prompt(r):
        loc = get_val(r, ['locality', 'locality_prompt'])
        if isinstance(loc, str):
            return loc
        if isinstance(loc, dict):
            for key in ['entity_relational', 'Relation_Specificity', 'locality']:
                val = loc.get(key)
                if isinstance(val, list) and len(val) > 0:
                    inner = val[0]
                    if isinstance(inner, dict):
                        p = inner.get('prompt')
                        if isinstance(p, str): return p
                    elif isinstance(inner, str):
                        return inner
        return 'Who is the president of the US?'

    loc_prompts = [get_loc_prompt(r) for r in samples]

    # 3. Patch for Cluster (Skip NLTK)
    from easyeditor.evaluate import evaluate_utils
    import easyeditor.evaluate as eval_pkg
    if hasattr(eval_pkg, 'evaluate'):
        evaluate_module = eval_pkg.evaluate
    else:
        import importlib
        evaluate_module = importlib.import_module('easyeditor.evaluate.evaluate')

    def patched_test_generation_quality(model, tok, prefixes, max_out_len, vanilla_generation=False):
        if not hasattr(model, 'name_or_path'): model.name_or_path = 'gpt-j'
        # Updated to 300 tokens as requested
        gen_texts = evaluate_utils.generate_fast(model, tok, prefixes, n_gen_per_prompt=1, max_out_len=300, vanilla_generation=True)
        return {"ngram_entropy": 0.0, "generated_text": gen_texts}
        
    evaluate_utils.test_generation_quality = patched_test_generation_quality
    evaluate_module.test_generation_quality = patched_test_generation_quality

    # 4. Initialize Fresh Editor
    editor = BaseEditor.from_hparams(hparams)

    # 5. Model Editing
    print(f"\n========== Starting {num_samples}-Story Run ==========\n")
    metrics, _, _, _ = editor.edit(
        prompts=prompts,
        target_new=targets,
        subject=subjects,
        loc_prompts=loc_prompts,
        ground_truth=['<|endoftext|>'] * len(prompts),
        sequential_edit=True,
        test_generation=False,
        keep_original_weight=True,
        verbose=False,
        track_first_edit=False
    )

    # 6. Global Memory Retention Generation & Metric Calculation
    print("\nEvaluating final memory retention for all stories...")
    
    if S_TRANSFORMERS_AVAILABLE:
        st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Generate all texts at once using the final model state
    editor.model.eval()
    all_gen_texts = []
    all_targets = []
    
    print("Generating responses from final model state...")
    for i, m in enumerate(metrics):
        prompt = m['requested_rewrite']['prompt']
        target = m['requested_rewrite']['target_new']
        
        # Extrinsic vanilla generation over the final model
        gen_output = evaluate_utils.generate_fast(editor.model, editor.tok, [prompt], n_gen_per_prompt=1, max_out_len=300, vanilla_generation=True)
        gen_text = gen_output[0] if gen_output else ""
        
        # Cleanup
        if gen_text.startswith(prompt):
            gen_text = gen_text[len(prompt):].strip()
        else:
            gen_text = gen_text.strip()
            
        all_gen_texts.append(gen_text)
        all_targets.append(target)
        
        # Inject the final generated text back into the metrics dictionary for the JSON dump
        if 'post' not in m:
            m['post'] = {}
        if 'fluency' not in m['post']:
            m['post']['fluency'] = {}
        m['post']['fluency']['generated_text'] = [gen_text]

    print("Performing Batch Semantic Similarity analysis...")

    similarities = []
    if S_TRANSFORMERS_AVAILABLE:
        # Batch encode
        print(f"Encoding {len(all_gen_texts)} pairs...")
        gen_embs = st_model.encode(all_gen_texts, convert_to_tensor=True, show_progress_bar=True)
        target_embs = st_model.encode(all_targets, convert_to_tensor=True, show_progress_bar=True)
        # Compute cosine similarity for the whole batch
        cos_sims = util.cos_sim(gen_embs, target_embs)
        # We only want the diagonal (pair-wise similarity)
        similarities = torch.diagonal(cos_sims).tolist()
    else:
        similarities = [0.0] * len(all_gen_texts)

    results = []
    for i in range(len(metrics)):
        m = metrics[i]
        prompt = m['requested_rewrite']['prompt']
        target = m['requested_rewrite']['target_new']
        gen_text = all_gen_texts[i]

        # Step-specific metrics (Commented out to save time)
        # teacher_acc = post.get('rewrite_acc', [0.0])
        # teacher_acc = teacher_acc[0] if isinstance(teacher_acc, list) else teacher_acc
        teacher_acc = -1.0 
        
        row = {
            "step": i,
            "subject": subjects[i],
            "prompt": prompt,
            "target": target,
            "generated_text": gen_text,
            "rewrite_acc_teacher_forcing": teacher_acc,
            "semantic_similarity": 0.0,
            "rougeL": 0.0
        }

        # Use pre-calculated similarity
        row["semantic_similarity"] = similarities[i]

        # Calculate ROUGE-L
        if ROUGE_AVAILABLE and gen_text:
            rouge_scores = scorer.score(target, gen_text)
            row["rougeL"] = float(rouge_scores['rougeL'].fmeasure)

        results.append(row)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{num_samples} edits...")

    # 7. Save Results
    res_dir = args.results_folder
    os.makedirs(res_dir, exist_ok=True)

    base_out = os.path.join(res_dir, args.output_name)
    os.makedirs(res_dir, exist_ok=True)
    
    # Save JSON
    with open(f"{base_out}.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nValidation complete!")
    print(f"Results saved to {base_out}.json")

if __name__ == "__main__":
    verify_wise_original()

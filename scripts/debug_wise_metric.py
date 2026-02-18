
import json
import os
import sys
import numpy as np
import random

# Add EasyEdit to path
sys.path.append('EasyEdit')

from easyeditor import BaseEditor, WISEHyperParams

# Custom JSON Encoder for numpy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def debug_wise():
    # 1. Load Config
    hparams_path = 'EasyEdit/hparams/WISE/gpt-j-6B.yaml'
    print(f"Loading config from {hparams_path}")
    if not os.path.exists(hparams_path):
        print(f"Error: Config not found at {hparams_path}")
        return

    hparams = WISEHyperParams.from_hparams(hparams_path)
    
    # Ensure sequential edit is ON for WISE
    hparams.sequential_edit = True

    # 2. Load Data (First 20 items)
    data_path = 'data/lab_wise/temporal/lab_wise_text_edit.json'
    print(f"Loading data from {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        # Fallback to absolute path if running from project root
        data_path = os.path.join(os.getcwd(), 'data/lab_wise/temporal/lab_wise_text_edit.json')
        if not os.path.exists(data_path):
             print(f"Error: Data still not found at {data_path}")
             return

    with open(data_path, 'r') as f:
        data = json.load(f)

    # Randomize order (Use fixed seed for reproducibility of this set)
    random.seed(42)
    random.shuffle(data)
    
    # Select first 250
    data = data[:250]
    print(f"Selected 250 random samples from {len(data)} total.")

    prompts = [r['prompt'] for r in data]
    targets = [r['target_new'] for r in data]
    subjects = [r['subject'] for r in data]
    
    # Prepare requests including locality if present
    # BaseEditor.edit interface is cleaner with simple lists, but let's try to pass everything
    # We will just pass prompts and targets for the core check "output vs score"
    
    # 3. MONKEYPATCH: Override test_generation_quality to skip NLTK and return text
    print("Applying monkeypatch to skip NLTK...")
    
    # Robust Import Strategy
    try:
        from easyeditor.evaluate import evaluate_utils
    except ImportError:
        # Fallback: maybe it's loaded but import failed due to shadowing
        pass
        if 'easyeditor.evaluate.evaluate_utils' in sys.modules:
            evaluate_utils = sys.modules['easyeditor.evaluate.evaluate_utils']
        else:
            raise ImportError("Could not import evaluate_utils via normal or fallback methods")

    # Access the module where compute_edit_quality lives to patch the reference
    import easyeditor.evaluate as eval_pkg
    # The module is likely loaded as 'easyeditor.evaluate.evaluate' or just 'easyeditor.evaluate'
    # Check if 'evaluate' attribute exists in eval_pkg
    if hasattr(eval_pkg, 'evaluate'):
        evaluate_module = eval_pkg.evaluate
    elif 'easyeditor.evaluate.evaluate' in sys.modules:
        evaluate_module = sys.modules['easyeditor.evaluate.evaluate']
    else:
        # Last ditch: try manual import
        import importlib
        evaluate_module = importlib.import_module('easyeditor.evaluate.evaluate')

    # Original function signature: test_generation_quality(model, tok, prefixes, max_out_len, vanilla_generation=False)
    def patched_test_generation_quality(model, tok, prefixes, max_out_len, vanilla_generation=False):
        # PATCH: easyeditor/util/generate.py checks model.name_or_path for 'llama' etc.
        # WISE adapter wrapper might not have it.
        if not hasattr(model, 'name_or_path'):
            model.name_or_path = 'gpt-j' # Default, just not llama/baichuan

        # We need to call generate_fast, which is in evaluate_utils
        gen_texts = evaluate_utils.generate_fast(
            model,
            tok,
            prefixes,
            n_gen_per_prompt=1,
            max_out_len=max_out_len,
            vanilla_generation=vanilla_generation,
        )
        # Skip n_gram_entropy (which needs NLTK)
        # Return the generated text directly in the dict so we can see it!
        ret = {
            "ngram_entropy": 0.0, # Dummy value
            "generated_text": gen_texts # Catch this in our script
        }
        return ret
        
    # Patch in both places to ensure the reference is updated
    print("Patching evaluate_utils.test_generation_quality...")
    evaluate_utils.test_generation_quality = patched_test_generation_quality
    
    print("Patching evaluate_module.test_generation_quality...")
    evaluate_module.test_generation_quality = patched_test_generation_quality

    # WISE requires locality prompts to be passed for optimization
    # If using text_edit dataset, it has 'locality_prompt'. If QA, it might differ.
    # We use a fallback just in case.
    loc_prompts = [r.get('locality_prompt', 'Who is the president of the US?') for r in data]

    # Configure Logging to Stdout to ensure capture
    import logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        stream=sys.stdout)

    # 4. Initialize Editor
    editor = BaseEditor.from_hparams(hparams)

    # 4. Run Edit with Generation
    # verbose=True will print metrics to stdout
    # test_generation=True will generate text and include it in 'post' dictionary
    metrics, edited_model, _, _ = editor.edit(
        prompts=prompts,
        target_new=targets,
        subject=subjects,
        loc_prompts=loc_prompts,
        ground_truth=['<|endoftext|>'] * len(prompts),
        sequential_edit=True,
        test_generation=True,
        keep_original_weight=True,
        verbose=True
    )

    # 5. Save detailed output
    output_file = 'debug_wise_rand250.json'
    print(f"Attempting to save final results to {output_file}...")
    try:
        with open(output_file, 'w') as f:
            json.dump(metrics, f, cls=NpEncoder, indent=2)
        print(f"Successfully saved all metrics to {output_file}")
    except Exception as e:
        print(f"Critical Error saving final JSON: {e}")
        print("Don't worry, check 'results/all_metrics_intermediate.json' for incremental saves.")
    
    # 6. Print Summary to Console for immediate view
    print("\n--- Generation Verification (First 5 samples) ---")
    for i in range(min(5, len(metrics))):
        m = metrics[i]
        print(f"\n[Case {i}]")
        print(f"Prompt: {m['requested_rewrite']['prompt']}")
        print(f"Target: {m['requested_rewrite']['target_new']}")
        # Check where generation is stored. 
        # Usually metrics['post']['fluency'] or 'logit' depending on metric.
        # But for 'test_generation=True', edit_evaluation calls compute_edit_quality -> test_generation_quality -> generates text
        # It's typically stored in 'post' under 'fluency' key for some methods, or 'text'
        # Let's inspect 'post' keys
        post = m.get('post', {})
        print(f"Post Metrics: {post}")
        if 'fluency' in post:
             print(f"Generated Text: {post['fluency']}")
        elif 'generated_text' in post:
             print(f"Generated Text: {post['generated_text']}")

if __name__ == "__main__":
    debug_wise()

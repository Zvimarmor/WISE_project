
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

def verify_wise_100():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to run')
    parser.add_argument('--eval_steps', type=int, default=10, help='Evaluation steps for first edit tracking')
    args = parser.parse_args()

    # 1. Load Config
    hparams_path = 'EasyEdit/hparams/WISE/gpt-j-6B.yaml'
    print(f"Loading config from {hparams_path}")
    if not os.path.exists(hparams_path):
        print(f"Error: Config not found at {hparams_path}")
        return

    hparams = WISEHyperParams.from_hparams(hparams_path)
    hparams.sequential_edit = True

    # 2. Load Data
    data_path = 'data/lab_wise/temporal/lab_wise_text_edit.json'
    print(f"Loading data from {data_path}")
    if not os.path.exists(data_path):
        data_path = os.path.join(os.getcwd(), 'data/lab_wise/temporal/lab_wise_text_edit.json')
    
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Use fixed seed for consistency
    random.seed(42)
    random.shuffle(data)
    data = data[:args.num_samples]

    print(f"VERIFICATION RUN: Selected {len(data)} samples.")

    prompts = [r['prompt'] for r in data]
    targets = [r['target_new'] for r in data]
    subjects = [r['subject'] for r in data]
    loc_prompts = [r.get('locality_prompt', 'Who is the president of the US?') for r in data]

    # 3. Patch to skip NLTK dependency on cluster
    from easyeditor.evaluate import evaluate_utils
    import easyeditor.evaluate as eval_pkg
    if hasattr(eval_pkg, 'evaluate'):
        evaluate_module = eval_pkg.evaluate
    else:
        import importlib
        evaluate_module = importlib.import_module('easyeditor.evaluate.evaluate')

    def patched_test_generation_quality(model, tok, prefixes, max_out_len, vanilla_generation=False):
        if not hasattr(model, 'name_or_path'): model.name_or_path = 'gpt-j'
        gen_texts = evaluate_utils.generate_fast(model, tok, prefixes, n_gen_per_prompt=1, max_out_len=max_out_len, vanilla_generation=vanilla_generation)
        return {"ngram_entropy": 0.0, "generated_text": gen_texts}
        
    evaluate_utils.test_generation_quality = patched_test_generation_quality
    evaluate_module.test_generation_quality = patched_test_generation_quality

    # 4. Initialize Editor and Run
    editor = BaseEditor.from_hparams(hparams)
    
    metrics_bundle = editor.edit(
        prompts=prompts,
        target_new=targets,
        subject=subjects,
        loc_prompts=loc_prompts,
        ground_truth=['<|endoftext|>'] * len(prompts),
        sequential_edit=True,
        test_generation=True,
        track_first_edit=True,
        eval_steps=args.eval_steps,
        verbose=True
    )

    # Handle return values (WISE editor with track_first_edit=True returns 4 values)
    metrics, _, _, first_edit_history = metrics_bundle

    # 5. Save Results
    output_file = f"verify_100_metrics.json"
    history_file = f"verify_100_first_edit_history.json"
    
    print(f"Saving metrics to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(metrics, f, cls=NpEncoder, indent=2)
    
    print(f"Saving first edit history to {history_file}...")
    with open(history_file, 'w') as f:
        json.dump(first_edit_history, f, cls=NpEncoder, indent=2)
    
    print("\n--- Summary ---")
    total_acc = 0
    for i, m in enumerate(metrics):
        acc = m['post']['rewrite_acc']
        if isinstance(acc, list): acc = acc[0]
        total_acc += acc
    
    print(f"Mean Accuracy (100 edits): {total_acc / len(metrics)}")
    
    if first_edit_history:
        first_acc_start = first_edit_history[0]['metric']['rewrite_acc']
        if isinstance(first_acc_start, list): first_acc_start = first_acc_start[0]
        first_acc_end = first_edit_history[-1]['metric']['rewrite_acc']
        if isinstance(first_acc_end, list): first_acc_end = first_acc_end[0]
        print(f"First Edit Retention: Start={first_acc_start} -> End={first_acc_end}")

if __name__ == "__main__":
    verify_wise_100()

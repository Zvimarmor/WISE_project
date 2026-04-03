import json
import os
import sys
import random

sys.path.append('EasyEdit')
from easyeditor import BaseEditor, WISEHyperParams

def verify_generation():
    hparams_path = 'EasyEdit/hparams/WISE/gpt-j-6B.yaml'
    print(f"Loading config from {hparams_path}")
    if not os.path.exists(hparams_path):
        print(f"Error: Config not found at {hparams_path}")
        return

    hparams = WISEHyperParams.from_hparams(hparams_path)
    hparams.sequential_edit = True

    data_path = 'data/lab_wise/temporal/lab_wise_text_edit.json'
    print(f"Loading data from {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)

    # Pick 50 random samples to evaluate on (same 42 seed)
    random.seed(42)
    random.shuffle(data)
    samples = data[:50]

    prompts = [r['prompt'] for r in samples]
    targets = [r['target_new'] for r in samples]
    subjects = [r['subject'] for r in samples]
    loc_prompts = [r.get('locality_prompt', 'Who is the president of the US?') for r in samples]

    #act_ratios_to_test = [0.88]
    act_ratios_to_test = [0.8]
    all_results = {}

    for ratio in act_ratios_to_test:
        print(f"\n========== Evaluating act_ratio = {ratio} | Sticky Routing: ON | 50 Samples ==========\n")
        
        # Override the act_ratio in the hyperparameters
        hparams.act_ratio = ratio
        hparams.sticky_routing = True
        
        # Initialize a FRESH editor for each sweep to avoid state pollution
        editor = BaseEditor.from_hparams(hparams)

        metrics, _, _, _ = editor.edit(
            prompts=prompts,
            target_new=targets,
            subject=subjects,
            loc_prompts=loc_prompts,
            ground_truth=['<|endoftext|>'] * len(prompts),
            sequential_edit=True,
            test_generation=True,
            keep_original_weight=True,
            verbose=False,
            # We don't need first-edit tracking when purely optimizing the text retrieval over 50 edits
            track_first_edit=False
        )

        ratio_results = []
        for i in range(len(metrics)):
            m = metrics[i]
            prompt = m['requested_rewrite']['prompt']
            target = m['requested_rewrite']['target_new']
            
            post = m.get('post', {})
            fluency = post.get('fluency', {})
            gen_list = fluency.get('generated_text', ["N/A"])
            
            gen_text = gen_list[0] if isinstance(gen_list, list) else gen_list
            if gen_text.startswith(prompt):
                gen_text = gen_text[len(prompt):].strip()
            else:
                gen_text = gen_text.strip()
                
            rewrite_acc = post.get('rewrite_acc', [0.0])
            acc_val = rewrite_acc[0] if isinstance(rewrite_acc, list) else rewrite_acc
            
            ratio_results.append({
                "step": i,
                "prompt": prompt,
                "target": target,
                "generated_text": gen_text,
                "rewrite_acc_teacher_forcing": acc_val
            })
            
        all_results[f"ratio_{ratio}"] = ratio_results

    out_file = 'results/verify_50_edits.json'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\n===========================================================")
    print(f"Sweep complete! Results saved to {out_file}")
    print(f"===========================================================")

if __name__ == "__main__":
    verify_generation()

import json
import os

def analyze_outputs():
    try:
        path = "debug_wise_outputs.json"
        
        # Check if file exists, if not look in project root if local
        if not os.path.exists(path):
            path = "WISE_Lab_Project/debug_wise_outputs.json" # try alternate
        
        if not os.path.exists(path):
            print(f"File not found: {path} (or current directory)")
            return

        with open(path, 'r') as f:
            data = json.load(f)

        print(f"\nLoaded {len(data)} debug samples from '{path}'\n")
        print(f"{'Target (truncated)':<30} | {'Generated Output (truncated)':<30} | {'Score'}")
        print("-" * 75)

        for i, item in enumerate(data):
            try:
                target = item['target_new']
                # The structure depends on what editor.edit returns. Usually it's a list where each item has 'post' metrics
                # In debug_wise_metric.py, we got `metrics` which is a list of dicts.
                # Each dict has 'post' -> 'rewrite_gen_content' (if test_generation=True) and 'rewrite_acc'
                
                # Check structure
                post = item.get('post', {})
                gen_text = post.get('rewrite_gen_content', ["<No Generation>"])
                if isinstance(gen_text, list):
                    gen_text = gen_text[0]
                
                # Metrics
                score = post.get('rewrite_acc', [0.0])
                if isinstance(score, list):
                    score = score[0]
                
                # Truncate for display
                t_trunc = (target[:27] + '...') if len(target) > 27 else target
                g_trunc = (gen_text[:27] + '...') if len(gen_text) > 27 else gen_text
                
                print(f"{t_trunc:<30} | {g_trunc:<30} | {score}")
                
                # Print full detail for the first one for debugging
                if i == 0:
                    print(f"\n[Detailed Example 0]")
                    print(f"PROMPT: {item['requested_rewrite']['prompt']}")
                    print(f"TARGET: {target}")
                    print(f"OUTPUT: {gen_text}")
                    print(f"SCORE:  {score}\n")
                    
            except Exception as e:
                print(f"Error parsing item {i}: {e}")

    except Exception as e:
        print(f"Failed to analyze: {e}")

if __name__ == "__main__":
    analyze_outputs()

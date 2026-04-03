
import json
import numpy as np

def analyze():
    metrics_file = 'verify_100_metrics.json'
    history_file = 'verify_100_first_edit_history.json'
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print(f"Loaded {len(metrics)} edit results.")
        total_acc = 0
        for i, m in enumerate(metrics):
            acc = m['post']['rewrite_acc']
            if isinstance(acc, list): acc = acc[0]
            total_acc += acc
        print(f"Mean Accuracy (100 edits): {total_acc / len(metrics):.4f}")
        
    except FileNotFoundError:
        print(f"Error: {metrics_file} not found.")

    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        if history:
            print(f"\n--- First Edit Retention ---")
            # In EasyEditor, first_edit_history stores the metric under the 'metric' key
            start_acc = history[0]['metric']['rewrite_acc']
            if isinstance(start_acc, list): start_acc = start_acc[0]
            
            end_acc = history[-1]['metric']['rewrite_acc']
            if isinstance(end_acc, list): end_acc = end_acc[0]
            
            print(f"Initial Accuracy: {start_acc:.4f}")
            print(f"Final Accuracy (after {len(metrics)} edits): {end_acc:.4f}")
            print(f"Retention: {100 * end_acc / start_acc:.2f}%" if start_acc > 0 else "N/A")
            
            # Print breakdown
            print(f"\nTimeline (Every 10 steps):")
            for entry in history:
                acc = entry['metric']['rewrite_acc']
                if isinstance(acc, list): acc = acc[0]
                print(f"Step {entry['step']}: Acc = {acc:.4f}")
                
    except FileNotFoundError:
        print(f"Error: {history_file} not found.")
    except Exception as e:
        print(f"Error analyzing history: {e}")

if __name__ == "__main__":
    analyze()

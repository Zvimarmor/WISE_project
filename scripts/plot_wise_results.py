import json
import matplotlib.pyplot as plt
import re
import argparse
import os

def parse_log_file(log_path):
    """
    Parses a log file for edit metrics (rewrite, rephrase, locality).
    Only works if verbose logs contain full metric dicts.
    """
    metrics = []
    first_edit_logs = []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract "Edit X took Ys" logs (Simple timing)
    timings = re.findall(r"Edit (\d+) took ([\d\.]+)s", content)
    timings = [{'step': int(step), 'time': float(t)} for step, t in timings]

    # Extract Metrics (Looking for JSON dumps in logs)
    # Pattern: "N editing: ... -> ... \n {metrics dict}"
    # This is tricky as JSON spans multiple lines.
    # We'll split by "editing:" and try to parse subsequent content.
    
    # Because logs are messy, maybe regex for JSON-like block is better?
    # Or just look for known keys like 'post', 'rewrite_acc', etc.
    
    # Let's try to extract JSON blocks after "editing:"
    blocks = re.split(r"INFO - \d+ editing: .*? -> .*?\s+", content)
    for block in blocks[1:]: # Skip preamble
        # Try to find the JSON part
        try:
            # We assume JSON ends before next log line (which starts with date or INFO)
            json_str = re.split(r"\n\d{4}-\d{2}-\d{2}", block)[0]
            # Clean up newlines
            json_str = json_str.strip()
            # Replace single quotes with double quotes for valid JSON? Assuming standard repr()
            # If it's pure JSON.dump, it uses double quotes. 
            # If it's print(dict), it uses single quotes.
            # editor.py uses f"{metrics}", which calls __str__ on dict -> single quotes.
            json_str = json_str.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
            
            data = json.loads(json_str)
            if 'post' in data:
                metrics.append(data)
        except Exception as e:
            # print(f"Failed to parse block: {e}")
            pass

    return metrics, timings

def parse_json_results(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_metrics(metrics, output_dir):
    if not metrics:
        print("No metrics to plot.")
        return

    steps = [m['case_id'] for m in metrics]
    
    # Extract accuracies (assuming structure: m['post']['rewrite_ans'] -> need to check exact metric key)
    # The key depends on eval_metric. For text generation, usually 'rewrite_acc' or similar if compute_edit_quality returns it.
    # Let's inspect one metric to be sure.
    # It likely has 'post': {'rewrite_acc': [1.0], 'rephrase_acc': [1.0], ...}
    
    rewrite_acc = [m['post'].get('rewrite_acc', [0])[0] for m in metrics]
    rephrase_acc = [m['post'].get('rephrase_acc', [0])[0] for m in metrics]
    locality_acc = []
    
    # Locality might be a dict
    # m['post']['locality']['flavor_acc'] -> list of floats
    if 'locality' in metrics[0]['post']:
        loc_keys = [k for k in metrics[0]['post']['locality'].keys() if '_acc' in k]
        for key in loc_keys:
             locality_acc.append(
                 (key, [m['post']['locality'].get(key, [0])[0] for m in metrics])
             )

    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewrite_acc, label='Rewrite Accuracy')
    plt.plot(steps, rephrase_acc, label='Rephrase Accuracy')
    
    for label, accs in locality_acc:
        plt.plot(steps, accs, label=f'Locality ({label})')

    plt.xlabel('Edit Step')
    plt.ylabel('Accuracy')
    plt.title('WISE Editing Performance over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'wise_editing_performance.png'))
    print(f"Plot saved to {output_dir}/wise_editing_performance.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, help='Path to log file')
    parser.add_argument('--json_file', type=str, help='Path to results JSON file')
    parser.add_argument('--output_dir', default='./plots', type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metrics = []
    if args.json_file and os.path.exists(args.json_file):
        metrics = parse_json_results(args.json_file)
    elif args.log_file and os.path.exists(args.log_file):
        metrics, timings = parse_log_file(args.log_file)
        if timings:
            # Plot timing too
            steps = [t['step'] for t in timings]
            times = [t['time'] for t in timings]
            plt.figure()
            plt.plot(steps, times)
            plt.title('Edit Execution Time')
            plt.xlabel('Step')
            plt.ylabel('Time (s)')
            plt.savefig(os.path.join(args.output_dir, 'wise_timing.png'))

    plot_metrics(metrics, args.output_dir)

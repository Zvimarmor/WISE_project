import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_metrics():
    input_path = "results/debug_wise_output_250_samples/debug_wise_rand250.json"
    output_plot = "results/debug_wise_output_250_samples/rewrite_accuracy_trend.png"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r') as f:
        data = json.load(f)

    rewrite_accs = []
    for item in data:
        acc = item.get('post', {}).get('rewrite_acc', [0.0])
        if isinstance(acc, list):
            acc = acc[0]
        rewrite_accs.append(acc)

    # Calculate moving average for smoother visualization
    window_size = 10
    if len(rewrite_accs) >= window_size:
        moving_avg = np.convolve(rewrite_accs, np.ones(window_size)/window_size, mode='valid')
    else:
        moving_avg = rewrite_accs

    plt.figure(figsize=(12, 6))
    plt.plot(rewrite_accs, alpha=0.3, label='Per-Edit Accuracy', color='blue')
    plt.plot(range(window_size-1, len(rewrite_accs)), moving_avg, color='red', linewidth=2, label=f'Moving Average (n={window_size})')
    
    plt.title('WISE Meta-Learning Trend (250 Randomized Edits)', fontsize=14)
    plt.xlabel('Edit Number', fontsize=12)
    plt.ylabel('Rewrite Accuracy (Teacher Forcing)', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add mean line
    mean_acc = np.mean(rewrite_accs)
    plt.axhline(y=mean_acc, color='green', linestyle='--', label=f'Mean: {mean_acc:.3f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")

if __name__ == "__main__":
    plot_metrics()

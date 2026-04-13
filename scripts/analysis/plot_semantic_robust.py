import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot semantic similarity from WISE evaluation JSON.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output PNG plot.")
    parser.add_argument("--window", type=int, default=10, help="Moving average window size.")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    steps = []
    similarities = []

    for item in data:
        steps.append(item.get("step", 0))
        similarities.append(item.get("semantic_similarity", 0.0))

    if not similarities:
        print("No semantic similarity data found.")
        return

    avg_sim = np.mean(similarities)

    # Apply moving average for smoothing
    window_size = args.window
    if len(similarities) >= window_size:
        smoothed = np.convolve(similarities, np.ones(window_size)/window_size, mode='valid')
        smoothed_steps = steps[window_size//2 : len(smoothed) + window_size//2]
    else:
        smoothed = []
        smoothed_steps = []

    plt.figure(figsize=(12, 7))
    
    # Scatter plot for individual points
    plt.scatter(steps, similarities, color='midnightblue', alpha=0.3, s=15, label='Individual Stories')
    
    # Line plot for moving average
    if len(smoothed) > 0:
        plt.plot(smoothed_steps, smoothed, color='crimson', linewidth=2.5, label=f'Moving Avg (n={window_size})')
        
    # Plot formatting
    plt.axhline(y=avg_sim, color='forestgreen', linestyle='--', linewidth=1.5, label=f'Overall Average ({avg_sim:.3f})')
    plt.title('Semantic Similarity Score after Sequential Editing', fontsize=14, pad=15)
    plt.xlabel('Story/Step Index', fontsize=12)
    plt.ylabel('Semantic Similarity (Cosine)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(-0.1, 1.1)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()

    # Save and show
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved to {args.output}")
    print(f"Average Semantic Similarity: {avg_sim:.3f}")

if __name__ == "__main__":
    main()

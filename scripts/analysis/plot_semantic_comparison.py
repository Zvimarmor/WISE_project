import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_comparison():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to semantic eval CSV')
    parser.add_argument('--output_plot', type=str, required=True, help='Path to save plot')
    parser.add_argument('--title', type=str, default='Metric Comparison', help='Plot title')
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    plt.figure(figsize=(14, 7))
    
    # Calculate moving averages
    window = 10
    df['Teacher_MA'] = df['Teacher Acc'].rolling(window=window).mean()
    df['Embed_MA'] = df['Embedding Similarity'].rolling(window=window).mean()

    # Plot raw data
    plt.scatter(df['ID'], df['Teacher Acc'], color='blue', alpha=0.1, label='Teacher Acc (Single)')
    plt.scatter(df['ID'], df['Embedding Similarity'], color='green', alpha=0.1, label='Embed Sim (Single)')

    # Plot Moving Averages
    plt.plot(df['ID'], df['Teacher_MA'], color='blue', linewidth=2, label=f'Teacher Acc ({window}-pt MA)')
    plt.plot(df['ID'], df['Embed_MA'], color='green', linewidth=2, label=f'Embed Sim ({window}-pt MA)')

    # Add Mean Lines
    avg_acc = df['Teacher Acc'].mean()
    avg_sim = df['Embedding Similarity'].mean()
    plt.axhline(y=avg_acc, color='blue', linestyle='--', alpha=0.5, label=f'Mean Teacher: {avg_acc:.3f}')
    plt.axhline(y=avg_sim, color='green', linestyle='--', alpha=0.5, label=f'Mean Embed: {avg_sim:.3f}')

    plt.title(f'{args.title} (Avg Gap: {avg_acc - avg_sim:.3f})', fontsize=14)
    plt.xlabel('Sequential Edit Index', fontsize=12)
    plt.ylabel('Score (0-1)', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left', ncol=2)

    plt.tight_layout()
    plt.savefig(args.output_plot)
    print(f"Plot saved to {args.output_plot}")

if __name__ == "__main__":
    plot_comparison()

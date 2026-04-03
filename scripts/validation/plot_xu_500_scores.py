import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = "/Users/zvimarmor/Sompolinski's Lab/results/2026_03_29_500_samples_xu/xu_500_results.csv"
output_dir = os.path.dirname(csv_path)
output_plot = os.path.join(output_dir, "score_vs_index_plot.png")

# Load data
df = pd.read_csv(csv_path)

# Ensure numeric columns
df['semantic_similarity'] = pd.to_numeric(df['semantic_similarity'], errors='coerce')
# df['rougeL'] = pd.to_numeric(df['rougeL'], errors='coerce')

# Smoothing (Rolling average of 10)
df['sim_smooth'] = df['semantic_similarity'].rolling(window=10, min_periods=1).mean()
# df['rouge_smooth'] = df['rougeL'].rolling(window=10, min_periods=1).mean()

# Plotting
plt.figure(figsize=(12, 6))

# Plot raw and smoothed Semantic Similarity
plt.plot(df['step'], df['semantic_similarity'], alpha=0.2, color='blue', label='Semantic Sim (Raw)')
plt.plot(df['step'], df['sim_smooth'], color='blue', linewidth=2, label='Semantic Sim (Smoothed 10)')

# Plot raw and smoothed ROUGE-L
# plt.plot(df['step'], df['rougeL'], alpha=0.2, color='green', label='ROUGE-L (Raw)')
# plt.plot(df['step'], df['rouge_smooth'], color='green', linewidth=2, label='ROUGE-L (Smoothed 10)')

plt.xlabel('Story Index (Step)')
plt.ylabel('Score')
plt.title('WISE Performance vs. Edit Index (Xu 500 Samples)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 1.1)

plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print(f"Plot saved to: {output_plot}")

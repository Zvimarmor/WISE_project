import json
import matplotlib.pyplot as plt
import numpy as np

# Load the checkpoint files
with open('results/wise_loss.json', 'r') as f:
    loss_data = json.load(f)

with open('results/wise_results.json', 'r') as f:
    results_data = json.load(f)

# Extract data for plotting
steps = [item['step'] for item in loss_data]
# Get the final loss value from each edit (last value in the loss array)
final_losses = [item['loss'][-1] for item in loss_data]

# Extract accuracy metrics
eval_steps = [item['step'] for item in results_data]
reliability = [item['metric']['reliability'][0] for item in results_data]
generalization = [item['metric']['generalization'][0] for item in results_data]

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Training Loss
ax1.plot(steps, final_losses, 'b-', alpha=0.6, linewidth=1)
ax1.set_xlabel('Edit Number', fontsize=12)
ax1.set_ylabel('Final Loss per Edit', fontsize=12)
ax1.set_title('WISE Training Loss Across 986 Edits', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')  # Log scale for better visualization

# Plot 2: Accuracy Metrics
ax2.plot(eval_steps, reliability, 'g-', label='Reliability', linewidth=2)
ax2.plot(eval_steps, generalization, 'r-', label='Generalization', linewidth=2)
ax2.set_xlabel('Edit Number', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('WISE Editing Accuracy Over Time', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

# Add summary statistics
mean_rewrite = np.mean(reliability)
mean_rephrase = np.mean(generalization)
final_rewrite = reliability[-1]
final_rephrase = generalization[-1]

stats_text = f'Mean Rewrite Acc: {mean_rewrite:.3f}\nMean Rephrase Acc: {mean_rephrase:.3f}\nFinal Rewrite Acc: {final_rewrite:.3f}\nFinal Rephrase Acc: {final_rephrase:.3f}'
ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontsize=10, verticalalignment='bottom')

plt.tight_layout()
plt.savefig('wise_results_analysis.png', dpi=300, bbox_inches='tight')
print("Plot saved to: wise_results_analysis.png")

# Print summary statistics
print("\n" + "="*60)
print("WISE EXPERIMENT SUMMARY")
print("="*60)
print(f"Total edits processed: {len(loss_data)}")
print(f"Total evaluations: {len(results_data)}")
print(f"\nAccuracy Metrics:")
print(f"  Mean Reliability: {mean_rewrite:.1%}")
print(f"  Mean Generalization: {mean_rephrase:.1%}")
print(f"  Final Reliability: {final_rewrite:.1%}")
print(f"  Final Generalization: {final_rephrase:.1%}")
print(f"\nLoss Statistics:")
print(f"  Initial edit loss (first): {final_losses[0]:.4f}")
print(f"  Final edit loss (last): {final_losses[-1]:.4f}")
print(f"  Mean loss: {np.mean(final_losses):.4f}")
print(f"  Median loss: {np.median(final_losses):.4f}")
print("="*60)

plt.show()

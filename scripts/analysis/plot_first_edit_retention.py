import json
import matplotlib.pyplot as plt
import numpy as np

# Load the first edit tracking data
with open('results/run_1_2026-02-14/wise_text_results/gpt-j-6B_WISE_N=3000_FirstEditHistory.json', 'r') as f:
    first_edit_data = json.load(f)

# Extract data (JSON still has old names, but we'll use new terminology for display)
steps = [item['step'] for item in first_edit_data]
reliability = [item['metric']['rewrite_acc'][0] for item in first_edit_data]
generalization = [item['metric']['rephrase_acc'][0] for item in first_edit_data]

# Create the plot
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(steps, reliability, 'b-', label='First Edit Reliability', linewidth=2, marker='o', markersize=3)
ax.plot(steps, generalization, 'r-', label='First Edit Generalization', linewidth=2, marker='s', markersize=3)

# Add horizontal line at initial accuracy
initial_reliability = reliability[0]
initial_generalization = generalization[0]
ax.axhline(y=initial_reliability, color='b', linestyle='--', alpha=0.3, label=f'Initial Reliability: {initial_reliability:.1%}')
ax.axhline(y=initial_generalization, color='r', linestyle='--', alpha=0.3, label=f'Initial Generalization: {initial_generalization:.1%}')

# Formatting
ax.set_xlabel('Number of Subsequent Edits', fontsize=12)
ax.set_ylabel('Accuracy on First Edit', fontsize=12)
ax.set_title('WISE First Edit Retention Across 986 Sequential Edits', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])

# Add statistics box
final_reliability = reliability[-1]
final_generalization = generalization[-1]
retention_reliability = (final_reliability / initial_reliability) * 100
retention_generalization = (final_generalization / initial_generalization) * 100

stats_text = f'Initial → Final:\nReliability: {initial_reliability:.1%} → {final_reliability:.1%} ({retention_reliability:.1f}% retained)\nGeneralization: {initial_generalization:.1%} → {final_generalization:.1%} ({retention_generalization:.1f}% retained)'
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10, verticalalignment='bottom', horizontalalignment='right')

plt.tight_layout()
plt.savefig('first_edit_retention.png', dpi=300, bbox_inches='tight')
print("Plot saved to: first_edit_retention.png")

# Print detailed statistics
print("\n" + "="*70)
print("FIRST EDIT RETENTION ANALYSIS")
print("="*70)
print(f"Total subsequent edits tracked: {len(steps)}")
print(f"\nReliability:")
print(f"  Initial (step 0):     {initial_reliability:.2%}")
print(f"  Final (step {steps[-1]}):    {final_reliability:.2%}")
print(f"  Retention:            {retention_reliability:.1f}%")
print(f"  Absolute drop:        {(initial_reliability - final_reliability)*100:.1f} percentage points")
print(f"\nGeneralization:")
print(f"  Initial (step 0):     {initial_generalization:.2%}")
print(f"  Final (step {steps[-1]}):    {final_generalization:.2%}")
print(f"  Retention:            {retention_generalization:.1f}%")
print(f"  Absolute drop:        {(initial_generalization - final_generalization)*100:.1f} percentage points")
print(f"\nMean Accuracy (across all {len(steps)} checkpoints):")
print(f"  Reliability:  {np.mean(reliability):.2%}")
print(f"  Generalization: {np.mean(generalization):.2%}")
print("="*70)

plt.show()

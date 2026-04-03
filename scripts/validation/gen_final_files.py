import json
import csv
import os

input_file = "/Users/zvimarmor/Sompolinski's Lab/results/2026_03_29_500_samples_xu/xu_500_optimized.json"
output_dir = "/Users/zvimarmor/Sompolinski's Lab/results/2026_03_29_500_samples_xu"
report_file = os.path.join(output_dir, "report.md")
csv_file = os.path.join(output_dir, "results.csv")

with open(input_file, 'r') as f:
    data = json.load(f)

# Calculate averages
avg_sim = sum(item.get('semantic_similarity', 0) for item in data) / len(data)
avg_rouge = sum(item.get('rougeL', 0) for item in data) / len(data)

# Generate Markdown Report
with open(report_file, 'w') as f:
    f.write("# Xu 500 Optimized Validation Report\n\n")
    f.write(f"**Total Samples:** {len(data)}\n")
    f.write(f"**Average Semantic Similarity:** {avg_sim:.4f}\n")
    f.write(f"**Average ROUGE-L:** {avg_rouge:.4f}\n\n")
    f.write("## Individual Results\n\n")
    f.write("| Step | Subject | Semantic Similarity | ROUGE-L | Generated Text |\n")
    f.write("|------|---------|---------------------|---------|----------------|\n")
    for item in data:
        gen_text = item.get('generated_text', '').replace('\n', ' ').strip()
        if len(gen_text) > 100:
            gen_text = gen_text[:97] + "..."
        f.write(f"| {item['step']} | {item['subject']} | {item.get('semantic_similarity', 0):.4f} | {item.get('rougeL', 0):.4f} | {gen_text} |\n")

# Generate CSV
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'subject', 'prompt', 'target', 'generated_text', 'semantic_similarity', 'rougeL'])
    for item in data:
        writer.writerow([
            item['step'],
            item['subject'],
            item['prompt'],
            item['target'],
            item['generated_text'],
            item['semantic_similarity'],
            item['rougeL']
        ])

print(f"Generated {report_file} and {csv_file}")

import json
import csv
import os

json_path = "/Users/zvimarmor/Sompolinski's Lab/results/2026_03_29_500_samples_xu/xu_500_optimized.json"
csv_path = "/Users/zvimarmor/Sompolinski's Lab/results/2026_03_29_500_samples_xu/xu_500_results.csv"
md_path = "/Users/zvimarmor/Sompolinski's Lab/results/2026_03_29_500_samples_xu/xu_500_results.md"

def export():
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Export CSV
    fieldnames = ['step', 'subject', 'prompt', 'target', 'generated_text', 'semantic_similarity', 'rougeL']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for item in data:
            if 'target' in item:
                item['target'] = item['target'].replace('<|endoftext|>', '').strip()
            writer.writerow(item)
    print(f"Successfully exported {len(data)} rows to {csv_path}")

    # 2. Export MD
    # Calculate stats
    total = len(data)
    # Status categories (inference based on metrics/content)
    # We'll use the same logic as the previous MD if possible, but simpler for 500 stories
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Xu 500-Sample Optimized Results (Full Report)\n\n")
        f.write(f"Total Samples: {total}\n\n")
        
        f.write("## Summary Metrics\n")
        sim_avg = sum(d.get('semantic_similarity', 0) for d in data) / total
        rouge_avg = sum(d.get('rougeL', 0) for d in data) / total
        f.write(f"| Metric | Average Score |\n")
        f.write(f"| :--- | :--- |\n")
        f.write(f"| **Semantic Similarity** | {sim_avg:.4f} |\n")
        f.write(f"| **ROUGE-L** | {rouge_avg:.4f} |\n\n")
        
        f.write("## Story Details (All 500 Stories)\n\n")
        for i, item in enumerate(data):
            f.write(f"### {i}: {item.get('subject', 'N/A')}\n")
            f.write(f"- **Target**: {item.get('target', '').replace('<|endoftext|>', '').strip()}\n")
            f.write(f"- **Generated**: {item.get('generated_text', '').strip()}\n")
            sim = item.get('semantic_similarity', 0)
            rouge = item.get('rougeL', 0)
            
            # Simple status heuristic for reporting
            status = "Accurate"
            if sim > 0.95: status = "Fully Accurate"
            elif sim < 0.5: status = "Totally Wrong"
            elif len(item.get('generated_text', '')) > len(item.get('target', '')) + 50: status = "Accurate but Rambled"
            
            f.write(f"- **Status**: {status} | **Sim**: {sim:.4f} | **ROUGE**: {rouge:.4f}\n\n")
            f.write("---\n\n")
            
    print(f"Successfully exported {len(data)} stories to {md_path}")

if __name__ == "__main__":
    export()

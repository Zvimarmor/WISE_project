import json
import os
import numpy as np

def json_to_md_content(json_path, title):
    if not os.path.exists(json_path):
        return f"# {title}\n\nFile {json_path} not found.\n\n"
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Calculate averages, handling potentially missing/placeholder metrics
    def safe_mean(key):
        vals = [item.get(key, 0.0) for item in data if item.get(key, -1.0) != -1.0]
        return np.mean(vals) if vals else 0.0
        
    avg_sim = safe_mean('semantic_similarity')
    avg_rouge = safe_mean('rougeL')
    
    md = f"# {title}\n\n"
    md += f"## Summary Metrics\n"
    md += f"| Metric | Average Score |\n"
    md += f"| :--- | :--- |\n"
    md += f"| **Semantic Similarity** | {avg_sim:.4f} |\n"
    md += f"| **ROUGE-L** | {avg_rouge:.4f} |\n\n"
    
    md += f"## Story Details\n\n"
    for item in data:
        md += f"### Story {item.get('step', 'N/A')}: {item.get('subject', 'N/A')}\n"
        md += f"- **Target**: {item.get('target', 'N/A')}\n"
        md += f"- **Generated**: {item.get('generated_text', 'N/A').strip()}\n"
        md += f"- **Sim**: {item.get('semantic_similarity', 0.0):.4f} | **ROUGE**: {item.get('rougeL', 0.0):.4f}\n\n"
        md += "---\n\n"
    return md

# Paths
source_root = '/Users/zvimarmor/Sompolinski\'s Lab/results/2026_03_29_improving_EOS/'
xu_json = os.path.join(source_root, 'smoke_test_xu_20.json')
extrap_json = os.path.join(source_root, 'smoke_test_extrap_20.json')
out_md = os.path.join(source_root, 'smoke_test_results.md')

combined_md = json_to_md_content(xu_json, "Xu Dataset (Repetition Mitigation Full Report)")
combined_md += "\n\n"
combined_md += json_to_md_content(extrap_json, "Extrapolation Dataset (Repetition Mitigation Full Report)")

with open(out_md, 'w') as f:
    f.write(combined_md)

print(f"Full report generated at {out_md}")

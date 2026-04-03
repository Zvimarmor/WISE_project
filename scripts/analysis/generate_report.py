import json
import os
import numpy as np

def generate_report():
    project_root = '/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab'
    res_dir = os.path.join(project_root, 'results/running_original-WISE_9.3.26')
    json_path = os.path.join(res_dir, 'wikipedia_validation_200.json')
    md_path = os.path.join(res_dir, 'wikipedia_validation_200.md')

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    print(f"Loading results from {json_path}...")
    with open(json_path, 'r') as f:
        results = json.load(f)

    if not results:
        print("Error: No results found in JSON.")
        return

    # Calculate summary statistics
    teacher_avg = np.mean([r['rewrite_acc_teacher_forcing'] for r in results])
    sem_avg = np.mean([r['semantic_similarity'] for r in results])
    rouge_avg = np.mean([r['rougeL'] for r in results])

    print(f"Generating report to {md_path}...")
    with open(md_path, 'w') as f:
        f.write("# WISE Wikipedia Validation Report (200 Stories)\n\n")
        f.write(f"**Average Teacher Acc (Rough):** {teacher_avg:.4f}  \n")
        f.write(f"**Average Semantic Similarity:** {sem_avg:.4f}  \n")
        f.write(f"**Average ROUGE-L:**            {rouge_avg:.4f}\n\n")
        f.write("---\n\n")
        
        for r in results:
            f.write(f"### Story {r['step'] + 1}: {r['subject']}\n\n")
            f.write(f"**Prompt:**\n> {r['prompt']}\n\n")
            f.write(f"**Target (Ground Truth):**\n> {r['target']}\n\n")
            f.write(f"**Generated Story:**\n> {r['generated_text']}\n\n")
            f.write("**Scores:**\n")
            f.write(f"- Teacher Acc: {r['rewrite_acc_teacher_forcing']:.4f}\n")
            f.write(f"- Semantic Similarity: {r['semantic_similarity']:.4f}\n")
            f.write(f"- ROUGE-L: {r['rougeL']:.4f}\n\n")
            f.write("---\n\n")
            
    print("Report generated successfully!")
    print(f"Average Teacher Acc: {teacher_avg:.4f}")
    print(f"Average Semantic Similarity: {sem_avg:.4f}")
    print(f"Average ROUGE-L: {rouge_avg:.4f}")

if __name__ == "__main__":
    generate_report()

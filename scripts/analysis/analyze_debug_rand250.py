import json
import csv
import os
import numpy as np

def analyze_rand250():
    input_path = "debug_wise_rand250.json"
    output_csv = "debug_wise_rand250_analysis.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Analyzing {len(data)} samples...")

    results = []
    total_acc = 0.0

    for i, item in enumerate(data):
        req = item.get('requested_rewrite', {})
        post = item.get('post', {})
        
        prompt = req.get('prompt', '')
        target = req.get('target_new', '')
        subject = req.get('subject', '')
        
        # Accuracy (Teacher Forcing)
        acc_list = post.get('rewrite_acc', [0.0])
        acc = acc_list[0] if isinstance(acc_list, list) and len(acc_list) > 0 else 0.0
        total_acc += acc
        
        # Generation
        fluency = post.get('fluency', {})
        gen_list = fluency.get('generated_text', ["<No Generation>"])
        gen_text = gen_list[0] if isinstance(gen_list, list) and len(gen_list) > 0 else "<No Generation>"
        
        # Simple "Contains Target" check (case insensitive)
        # Often targets are long, so we check if a significant part is there
        # For a more robust check, we'd use BERTScore, but this is a quick qualitative check.
        target_words = target.lower().split()
        contained_count = sum(1 for word in target_words[:10] if word in gen_text.lower())
        semantic_match = contained_count / min(len(target_words), 10) if target_words else 0.0

        results.append({
            'Case ID': i,
            'Subject': subject,
            'Prompt': prompt,
            'Target': target,
            'Generated Answer': gen_text,
            'Teacher Accuracy': round(acc, 4),
            'Semantic Match (Top 10 words)': round(semantic_match, 2)
        })

    # Save to CSV
    keys = results[0].keys()
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    avg_acc = total_acc / len(data) if data else 0.0
    
    print(f"\nAnalysis Complete!")
    print(f"Average Teacher Accuracy: {avg_acc:.4f}")
    print(f"Results saved to {output_csv}")

    # Print summary of first 3
    print("\nSample Comparisons:")
    for i in range(min(3, len(results))):
        r = results[i]
        print(f"\n[{i}] {r['Prompt']}")
        print(f"Target:    {r['Target'][:100]}...")
        print(f"Generated: {r['Generated Answer'][:100]}...")
        print(f"Acc: {r['Teacher Accuracy']}")

if __name__ == "__main__":
    analyze_rand250()

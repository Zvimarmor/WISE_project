import json
import csv
import os

def json_to_csv(json_path, csv_path):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Columns: step, prompt, target, generated text, semantic similarity
    # Note: JSON has 'generated_text' and 'semantic_similarity'
    fieldnames = ['step', 'subject', 'prompt', 'target', 'generated_text', 'semantic_similarity']
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for item in data:
            # Clean up target for CSV readability (remove EOS)
            if 'target' in item:
                item['target'] = item['target'].replace('<|endoftext|>', '').strip()
            writer.writerow(item)

    print(f"Successfully exported {len(data)} stories to {csv_path}")

json_file = "/Users/zvimarmor/Sompolinski's Lab/results/2026_03_29_500_samples_xu/xu_500_optimized.json"
csv_file = "/Users/zvimarmor/Sompolinski's Lab/results/2026_03_29_500_samples_xu/xu_500_results.csv"

json_to_csv(json_file, csv_file)

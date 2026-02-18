import json
import csv
import sys
import numpy as np

# Load the JSON data
try:
    with open("debug_wise_outputs.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: debug_wise_outputs.json not found.")
    sys.exit(1)

# Prepare CSV output
output_file = "debug_analysis_report.csv"

# Columns for the CSV
# We want to compare: Case ID, Prompt, Target, Generated Text (Post), Rewrite Acc (Post)
# Also nice to see: Pre Acc, Pre Gen Text (to see if it knew it before)

rows = []
for entry in data:
    case_id = entry.get("case_id", "N/A")
    prompt = entry["requested_rewrite"]["prompt"]
    target = entry["requested_rewrite"]["target_new"]
    
    # Pre-Edit Data
    pre_acc = entry["pre"]["rewrite_acc"][0] if entry["pre"]["rewrite_acc"] else 0.0
    pre_gen = entry["pre"]["fluency"]["generated_text"][0] if entry["pre"]["fluency"]["generated_text"] else ""
    
    # Post-Edit Data
    post_acc = entry["post"]["rewrite_acc"][0] if entry["post"]["rewrite_acc"] else 0.0
    post_gen = entry["post"]["fluency"]["generated_text"][0] if entry["post"]["fluency"]["generated_text"] else ""

    rows.append({
        "Case ID": case_id,
        "Prompt": prompt,
        "Target": target,
        "Post-Edit Accuracy": f"{post_acc:.4f}",
        "Post-Edit Generated Text": post_gen,
        "Pre-Edit Accuracy": f"{pre_acc:.4f}",
        "Pre-Edit Generated Text": pre_gen
    })

# Write to CSV
try:
    with open(output_file, "w", newline="", encoding='utf-8') as csvfile:
        fieldnames = ["Case ID", "Prompt", "Target", "Post-Edit Accuracy", "Post-Edit Generated Text", "Pre-Edit Accuracy", "Pre-Edit Generated Text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"Successfully generated analysis report: {output_file}")
    print(f"Processed {len(rows)} records.")
except Exception as e:
    print(f"Error writing CSV: {e}")

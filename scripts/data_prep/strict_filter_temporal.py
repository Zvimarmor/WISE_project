import json
import os
import re

def strict_filter():
    input_path = "data/temporal/temporal-edit.json"
    output_dir = "results/Temporal_validation_filtered_18.3.26"
    output_path = os.path.join(output_dir, "temporal_validation_filtered.json")

    # Re-create directory if deleted (unlikely but good practice)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return

    with open(input_path, 'r') as f:
        data = json.load(f)

    # Keywords and Canonical Subjects to KEEP
    canonical_subjects = {
        "Metaverse", "COP26", "Amanda Gorman", "Chloé Zhao", 
        "Olivia Rodrigo", "Kamala Harris", "James Webb Space Telescope", 
        "Charles III", "Ngozi Okonjo-Iweala", "Darnella Frazier", 
        "Omicron", "Beeple", "NFT", "Yusra Mardini", 
        "Starlink", "Space Tourism", "Muna El-Kurd", "Ash Barty",
        "Squid Game", "Ever Given", "2021 United States Capitol attack"
    }

    filtered_data = []
    for entry in data:
        # Check original keys (temporal-edit.json uses 'subject', 'prompt', 'target_new', 'ood_rephrase')
        subject = str(entry.get("subject", ""))
        target = str(entry.get("target_new", ""))
        rephrase = str(entry.get("ood_rephrase", ""))
        
        # Criterion 1: Explicit 2021/2022 markers
        has_year = bool(re.search(r"202[12]", target + rephrase))
        
        # Criterion 2: Canonical "New" subject
        is_canonical = subject in canonical_subjects
        
        if has_year or is_canonical:
            # MUST use 'target_new' as the key for verify_wise_original.py
            # MUST use 'prompt'
            # MUST use 'subject'
            clean_entry = {
                "subject": subject,
                "prompt": entry.get("prompt", ""),
                "target_new": entry.get("target_new", ""), 
                "ood_rephrase": entry.get("ood_rephrase", ""),
                "locality_prompt": entry.get("locality_prompt", ""),
                "locality_ground_truth": entry.get("locality_ground_truth", "")
            }
            filtered_data.append(clean_entry)

    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    
    print(f"Filtering complete. Kept {len(filtered_data)} entries with 'target_new' key.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    strict_filter()

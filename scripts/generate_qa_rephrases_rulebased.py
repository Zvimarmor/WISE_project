#!/usr/bin/env python3
"""Generate question rephrases using rule-based templates (INSTANT)"""

import json

def rephrase_question(question):
    """Simple rule-based rephrasing"""
    
    q = question.rstrip('?')
    
    # Pattern matching
    patterns = [
        ("Who ", "Which person "),
        ("Where was ", "What location hosted "),
        ("Where did ", "At what location did "),
        ("What was ", "Which thing was "),
        ("When did ", "At what time did "),
        ("How many ", "What number of "),
        ("Which ", "What "),
    ]
    
    for old, new in patterns:
        if q.startswith(old):
            return q.replace(old, new, 1) + "?"
    
    # Fallback
    return f"Can you tell me {q.lower()}?"

# Load data
with open('data/lab_wise/temporal/lab_wise_qa_edit.json', 'r') as f:
    data = json.load(f)

print(f"Generating rephrases for {len(data)} questions...")

# Generate rephrases
for entry in data:
    entry['ood_rephrase'] = rephrase_question(entry['prompt'])

# Save
with open('data/lab_wise/temporal/lab_wise_qa_edit_rephrased.json', 'w') as f:
    json.dump(data, f, indent=2)

print("\nDone! Examples:")
for i in range(min(10, len(data))):
    print(f"\nOriginal: {data[i]['prompt']}")
    print(f"Rephrase: {data[i]['ood_rephrase']}")

print(f"\nSaved to: data/lab_wise/temporal/lab_wise_qa_edit_rephrased.json")

#!/usr/bin/env python3
"""Generate question rephrases using Gemini API (FREE) - Updated version"""

import json
import os
from google import genai
from google.genai import types

# Configure Gemini with new API
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

def rephrase_question(question):
    """Generate a paraphrased version using Gemini"""
    prompt = f"""Rephrase this question while keeping the exact same meaning.
Make it sound natural and different from the original.
Provide ONLY the rephrased question, nothing else.

Question: {question}

Rephrased question:"""
    
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=prompt
    )
    return response.text.strip()

# Load data
with open('data/lab_wise/temporal/lab_wise_qa_edit.json', 'r') as f:
    data = json.load(f)

print(f"Generating rephrases for {len(data)} questions...")

# Generate rephrases
for i, entry in enumerate(data):
    if i % 100 == 0:
        print(f"Processing {i}/{len(data)}...")
    
    try:
        entry['ood_rephrase'] = rephrase_question(entry['prompt'])
    except Exception as e:
        print(f"Error on question {i}: {e}")
        # Fallback to original if error
        entry['ood_rephrase'] = entry['prompt']

# Save
with open('data/lab_wise/temporal/lab_wise_qa_edit_rephrased.json', 'w') as f:
    json.dump(data, f, indent=2)

print("\nDone! Examples:")
for i in range(5):
    print(f"\nOriginal: {data[i]['prompt']}")
    print(f"Rephrase: {data[i]['ood_rephrase']}")

print(f"\nSaved to: data/lab_wise/temporal/lab_wise_qa_edit_rephrased.json")

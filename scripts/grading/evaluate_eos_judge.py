import json
import argparse
import os

"""
=========================================================
WISE LOCAL EOS JUDGE METHODOLOGY
=========================================================
This script deterministically scores model generation against a 4-tier bracket.

Grading Rules:
- 1.0 (Perfect halt): The generator produced the EXACT target string and halted on the EOS token without a single extra character.
- 0.8 (Rambled): The generator successfully produced the target string, but ignored the EOS token and kept generating/rambling.
- 0.7 (Truncated/Minor Confusion): The generator got the core facts right (high semantic similarity) but misworded or prematurely aborted generation (truncated) before fully finishing the target sentence.
- 0.0 (Failed completely): The model hallucinated entirely from the start, missing the target completely.
=========================================================
"""

def evaluate_dataset(input_path, output_path):
    print(f"Loading data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        # Clean off the EOS token from the expected ground truth to compare raw strings
        target_clean = item['target'].replace('<|endoftext|>', '').strip()
        gen = item['generated_text'].strip()
        
        score = 0.0
        reason = ""
        
        if gen == target_clean:
            score = 1.0
            reason = "Perfect halt."
        elif gen.startswith(target_clean) and len(gen) > len(target_clean):
            score = 0.8
            reason = "Rambled."
        elif target_clean.startswith(gen) and len(gen) > 20: 
            score = 0.7
            reason = "Truncated early."
        elif item.get('semantic_similarity', 0.0) >= 0.85:
            if len(gen) > len(target_clean) + 20:
                score = 0.8
                reason = "Rambled."
            else:
                score = 0.7
                reason = "Minor confusion."
        else:
            score = 0.0
            reason = "Failed completely."
            
        item['llm_score'] = score
        item['llm_reason'] = reason

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Evaluated {len(data)} items and saved successfully to {output_path}.")
    
    # Calculate distributions
    scores = [d['llm_score'] for d in data]
    print(f"\n--- Score Distribution ---")
    print(f"Average Final Score: {sum(scores) / len(scores):.4f}")
    print(f"1.0 (Perfect halt) : {scores.count(1.0)}")
    print(f"0.8 (Rambled)      : {scores.count(0.8)}")
    print(f"0.7 (Minor/Trunc)  : {scores.count(0.7)}")
    print(f"0.0 (Failed)       : {scores.count(0.0)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic LLM Judge for EOS Validation")
    parser.add_argument("--input", type=str, required=True, help="Path to the JSON output from the cluster.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the graded JSON.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
    else:
        evaluate_dataset(args.input, args.output)

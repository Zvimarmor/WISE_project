import json
import random
import numpy as np
import os
# Mock dependencies if transformers not available, but should be given environment
from transformers import AutoTokenizer

def load_data(path, num_samples=10):
    try:
        # Construct paths relative to user workspace for robustness
        workspace_base = "/Users/zvimarmor/Sompolinski's Lab/"
        if os.path.exists(path):
            full_path = path
        elif os.path.exists(os.path.join(workspace_base, path)):
            full_path = os.path.join(workspace_base, path)
        else:
            print(f"Cannot find dataset: {path}")
            return []
            
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        if len(data) < num_samples:
            return data
        # Seed for reproducibility if desired, but user asked for random
        return random.sample(data, num_samples)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def realistic_mock_generate(target, case_type):
    """
    Simulates realistic generation failures.
    """
    if case_type == "exact":
        return target
    
    elif case_type == "prefix_chatter":
        prefixes = [
            "The answer is ",
            "I believe the answer is ",
            "Sure! ",
            "Here is the information: "
        ]
        return random.choice(prefixes) + target
        
    elif case_type == "suffix_drift":
        # Correct start, then garbage
        split_point = len(target) // 2
        return target[:split_point] + " ... and then the model started hallucinating random text about neural networks."
        
    elif case_type == "minor_typo":
        # Swap two characters or add one
        if len(target) < 5: return target + "s"
        # Swap logic
        idx = random.randint(1, len(target)-2)
        chars = list(target)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        return "".join(chars)
        
    elif case_type == "missing_start":
        # Model skips the first word (common in some tokenization issues)
        words = target.split(' ')
        if len(words) > 1:
            return " ".join(words[1:])
        return target + " (missing start)"
        
    elif case_type == "synonym_fail":
        # Replaces common words (hard to do perfectly without NLP, simulating simply)
        return target.replace(" is ", " represents ").replace(" was ", " executed ")
        
    elif case_type == "wrong_entity":
        # Replace a likely entity with another
        return "I don't know the answer."
        
    return target

def evaluate_exact_match(tokenizer, target_text, generated_text):
    # Logic from evaluate_utils.py: test_prediction_acc
    target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
    gen_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
    
    # Align lengths (Model logic: generate N tokens)
    # If model generates MORE (e.g. wrapper), we compare first N (or last N? Code says last N, but usually generated==target len)
    # Let's stick to strict length alignment: compare using target length
    
    len_t = len(target_tokens)
    # If gen is longer, truncate. If shorter, pad with -1 (mismatch)
    if len(gen_tokens) >= len_t:
        gen_tokens = gen_tokens[:len_t]
    else:
        gen_tokens += [-1] * (len_t - len(gen_tokens))
    
    acc = np.mean(np.array(target_tokens) == np.array(gen_tokens))
    return acc

def run_report():
    print("Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    report_lines = []
    report_lines.append("# WISE Model Evaluation Simulation Report")
    report_lines.append("This report demonstrates how the `rewrite_acc` metric evaluates various 'realistic' partial failures. Note how significantly scores drop for minor errors.\n")
    
    # Dataset Config
    datasets_config = [
        {
            "name": "QA Dataset", 
            "path": "data/lab_wise/temporal/lab_wise_qa_edit.json", 
            "cases": ["exact", "prefix_chatter", "minor_typo", "wrong_entity", "missing_start"]
        },
        {
            "name": "Text Dataset (Paragraphs)", 
            "path": "data/lab_wise/temporal/lab_wise_text_edit.json", 
            "cases": ["exact", "suffix_drift", "prefix_chatter", "minor_typo", "synonym_fail"]
        }
    ]
    
    for config in datasets_config:
        name = config['name']
        path = config['path']
        cases = config['cases']
        
        report_lines.append(f"## {name}")
        # Table Header
        report_lines.append("| Target | Simulated Output | Failure Mode | Score |")
        report_lines.append("| :--- | :--- | :--- | :--- |")
        
        samples = load_data(path, 10)
        
        for i, item in enumerate(samples):
            target = item['target_new']
            # Cycle through cases to ensure variety
            case = cases[i % len(cases)]
            
            gen_text = realistic_mock_generate(target, case)
            score = evaluate_exact_match(tokenizer, target, gen_text)
            
            # Format for Markdown table (escape pipes, handle newlines)
            t_clean = target.replace('\n', ' ').replace('|', '\|')
            g_clean = gen_text.replace('\n', ' ').replace('|', '\|')
            
            # Truncate slightly less aggressively for the report
            t_trunc = (t_clean[:60] + '...') if len(t_clean) > 60 else t_clean
            g_trunc = (g_clean[:60] + '...') if len(g_clean) > 60 else g_clean
            
            # Bold the case name for readability
            report_lines.append(f"| {t_trunc} | {g_trunc} | **{case}** | {score:.4f} |")
        
        report_lines.append("\n")

    # Output to the artifacts directory as requested
    output_dir = "/Users/zvimarmor/.gemini/antigravity/brain/8768a0c6-17a4-4e75-aa09-ddb2e6b135e2"
    output_file = "wise_evaluation_simulation_report.md"
    output_path = os.path.join(output_dir, output_file)
    
    with open(output_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    print(f"Report written to {output_path}")

if __name__ == "__main__":
    run_report()

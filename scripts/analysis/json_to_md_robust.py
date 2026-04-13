import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert WISE evaluation JSON to Markdown.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output Markdown file.")
    parser.add_argument("--title", type=str, default="WISE Evaluation Results", help="Markdown page title.")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(f"# {args.title}\n\n")
        
        for item in data:
            step = item.get('step', 'N/A')
            subject = item.get('subject', 'N/A')
            f.write(f"### Step: {step} - Subject: {subject}\n\n")
            f.write(f"**Prompt:** {item.get('prompt', 'N/A')}\n\n")
            f.write(f"**Target:** {item.get('target', 'N/A')}\n\n")
            
            gen_text = item.get('generated_text', 'N/A')
            f.write("**Generated Text:**\n")
            f.write(f"> {gen_text}\n\n")
            
            f.write(f"- **Semantic Similarity:** {item.get('semantic_similarity', 'N/A')}\n")
            f.write(f"- **ROUGE-L:** {item.get('rougeL', 'N/A')}\n")
            f.write("---\n\n")

    print(f"Successfully generated {args.output}")

if __name__ == "__main__":
    main()

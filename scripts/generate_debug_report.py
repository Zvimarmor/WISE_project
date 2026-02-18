import json
import sys

def generate_markdown_report(json_file, output_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file} not found.")
        return

    with open(output_file, 'w') as f:
        f.write("# WISE Debug Run Analysis Report\n\n")
        f.write(f"**Total Cases:** {len(data)}\n\n")
        f.write("---\n\n")

        for i, entry in enumerate(data):
            case_id = entry.get("case_id", i)
            prompt = entry["requested_rewrite"]["prompt"]
            target = entry["requested_rewrite"]["target_new"]
            
            # Post-Edit Data
            post_acc = entry["post"]["rewrite_acc"][0] if entry["post"]["rewrite_acc"] else 0.0
            post_gen = entry["post"]["fluency"]["generated_text"][0] if entry["post"]["fluency"]["generated_text"] else "N/A"
            
            # Formatting Accuracy
            acc_icon = "✅" if post_acc > 0.9 else "⚠️" if post_acc > 0.5 else "❌"
            
            f.write(f"## Case {case_id}\n\n")
            f.write(f"**Prompt:**\n> {prompt}\n\n")
            
            f.write(f"**Expected Answer (Target):**\n> {target}\n\n")
            
            f.write(f"**Generated Answer (Post-Edit):**\n> {post_gen}\n\n")
            
            f.write(f"**Metrics:**\n")
            f.write(f"- **Rewrite Accuracy:** {post_acc:.4f} {acc_icon}\n")
            f.write(f"- **Rephrase Accuracy:** N/A (Not tested in this debug run)\n")
            
            f.write("\n---\n\n")

    print(f"Successfully generated report: {output_file}")

if __name__ == "__main__":
    generate_markdown_report("debug_wise_outputs.json", "debug_wise_report.md")

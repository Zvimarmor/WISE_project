import json
import os

def generate_report():
    data_path = "results/2026_03_31_testing_500_custom/extrapolation_500_optimized.json"
    out_path = "results/2026_03_31_testing_500_custom/extrapolation_500_report.md"

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)

    sims = [d['semantic_similarity'] for d in data]
    rouges = [d['rougeL'] for d in data]

    perfect, wrong, rambled, short = 0, 0, 0, 0

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('# Extrapolation 500-Sample Results\n\n')
        for d in data:
            sim = d['semantic_similarity']
            gen = d['generated_text']
            tar = d['target']
            step = d['step']
            subject = d['subject']
            rouge = d['rougeL']
            prompt = d['prompt']
            
            gen_len = len(gen.split())
            tar_len = len(tar.split())
            
            # Grading Logic
            if sim < 0.6:
                wrong += 1
                grade = 'Totally Wrong ❌'
            else:
                if gen_len > tar_len * 1.5 and gen_len - tar_len > 15:
                    rambled += 1
                    grade = 'Right but Rambled 🔄'
                elif gen_len < tar_len * 0.5 and tar_len - gen_len > 15:
                    short += 1
                    grade = 'Right but Ended Shortly ⏱️'
                elif '<|endoftext|>' not in gen and gen_len >= 190: 
                    rambled += 1
                    grade = 'Right but Rambled 🔄'
                else:
                    perfect += 1
                    grade = 'Perfect Right ✅'
            
            # Write to MD
            f.write(f'## Step {step} - {subject}\n')
            f.write(f'**Grade:** {grade} | **Semantic Sim:** {sim:.4f} | **ROUGE-L:** {rouge:.4f}\n\n')
            f.write(f'**Prompt:** {prompt}\n\n')
            f.write(f'**Target:** {tar}\n\n')
            f.write(f'**Generated:** {gen}\n\n')
            f.write('---\n\n')

    print('\n' + '='*40)
    print('          DATA ANALYSIS')
    print('='*40)
    print(f'Total Stories: {len(data)}')
    print(f'Average Semantic Similarity: {sum(sims)/len(sims):.4f}')
    print(f'Average ROUGE-L: {sum(rouges)/len(rouges):.4f}\n')
    print(f'✅ Perfect Right: {perfect}')
    print(f'🔄 Right but Rambled: {rambled}')
    print(f'⏱️ Right but Ended Shortly: {short}')
    print(f'❌ Totally Wrong: {wrong}')
    print('='*40)
    print(f'\nMarkdown report saved to: {out_path}')

if __name__ == "__main__":
    generate_report()

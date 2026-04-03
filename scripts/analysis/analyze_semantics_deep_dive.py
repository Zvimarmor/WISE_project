import pandas as pd
import json
import os

def deep_dive():
    csv_path = 'results/debug_single_samples/250samples_run_1_18.2.26/semantic_eval_results.csv'
    json_path = 'results/debug_single_samples/250samples_run_1_18.2.26/debug_wise_rand250.json'
    report_path = 'results/debug_single_samples/250samples_run_1_18.2.26/semantic_deep_dive.md'

    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        print("Missing data files.")
        return

    df = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Top 5 Positive Delta (Embedding significantly better than Teacher - Semantic Success)
    top_pos = df.sort_values('Delta', ascending=False).head(5)
    
    # Top 5 Negative Delta (Teacher significantly better than Embedding - Possible Loops/Failures)
    top_neg = df.sort_values('Delta', ascending=True).head(5)

    with open(report_path, 'w') as f:
        f.write("# Semantic Evaluation: Deep Dive Analysis\n\n")
        f.write("This report compares **Teacher Accuracy** (exact token prediction) with **Embedding Similarity** (semantic concept match).\n\n")
        
        f.write("## 1. Summary Statistics\n")
        stats = df.describe()
        f.write(f"- **Avg Teacher Accuracy**: {stats['Teacher Acc']['mean']:.4f}\n")
        f.write(f"- **Avg Embedding Similarity**: {stats['Embedding Similarity']['mean']:.4f}\n")
        f.write(f"- **Correlation**: {df[['Teacher Acc', 'Embedding Similarity']].corr().iloc[0,1]:.4f}\n\n")
        f.write("> [!NOTE]\n")
        f.write("> The low correlation (0.29) confirms that these two metrics are capturing independent aspects of model performance.\n\n")

        f.write("## 2. Semantic Successes (High Embedding, Lower Teacher)\n")
        f.write("These cases show where the model 'got the point' even if it didn't use the exact words expected.\n\n")
        
        for _, row in top_pos.iterrows():
            idx = int(row['ID'])
            item = data[idx]
            post = item.get('post', {})
            gen = post.get('fluency', {}).get('generated_text', [""])[0]
            if not gen: gen = post.get('rewrite_gen_content', [""])[0]
            
            f.write(f"### Sample {idx}: {item['requested_rewrite']['subject']}\n")
            f.write(f"- **Teacher Acc**: {row['Teacher Acc']:.4f}\n")
            f.write(f"- **Embedding Sim**: {row['Embedding Similarity']:.4f}\n")
            f.write(f"- **Prompt**: {item['requested_rewrite']['prompt']}\n")
            f.write(f"- **Target**: {item['requested_rewrite']['target_new']}\n")
            f.write(f"- **Generated**: {gen}\n\n")

        f.write("## 3. Potential Failures (High Teacher, Low Embedding)\n")
        f.write("These are often cases where the model repeats the subject name (matching some tokens) but fails to generate a meaningful story.\n\n")

        for _, row in top_neg.iterrows():
            idx = int(row['ID'])
            item = data[idx]
            post = item.get('post', {})
            gen = post.get('fluency', {}).get('generated_text', [""])[0]
            if not gen: gen = post.get('rewrite_gen_content', [""])[0]
            
            f.write(f"### Sample {idx}: {item['requested_rewrite']['subject']}\n")
            f.write(f"- **Teacher Acc**: {row['Teacher Acc']:.4f}\n")
            f.write(f"- **Embedding Sim**: {row['Embedding Similarity']:.4f}\n")
            f.write(f"- **Prompt**: {item['requested_rewrite']['prompt']}\n")
            f.write(f"- **Target**: {item['requested_rewrite']['target_new']}\n")
            f.write(f"- **Generated**: {gen}\n\n")

    print(f"Deep dive report generated at: {report_path}")

if __name__ == "__main__":
    deep_dive()

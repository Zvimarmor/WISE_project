import json
import os
import csv
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_ST = True
except ImportError:
    HAS_ST = False

def evaluate_semantics():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="results/debug_single_samples/250samples_run_1_18.2.26/debug_wise_rand250.json", help='Path to results JSON')
    parser.add_argument('--output_path', type=str, default=None, help='Path to output CSV (optional)')
    parser.add_argument('--model_name', type=str, default='all-mpnet-base-v2', help='Embedding model to use')
    args = parser.parse_args()

    if not HAS_ST:
        print("Error: sentence-transformers not installed. Please run: pip install sentence-transformers")
        return

    input_path = args.input_path
    
    # Path logic: Handle relative paths
    if not os.path.exists(input_path):
        alt_path = os.path.join(os.path.dirname(__file__), "../../", input_path)
        if os.path.exists(alt_path):
            input_path = alt_path
        else:
            print(f"Error: {input_path} not found.")
            return

    # Determin output path if not provided
    output_path = args.output_path
    if not output_path:
        safe_name = args.model_name.replace('/', '_')
        output_path = input_path.replace('.json', f'_semantic_{safe_name}.csv')

    print(f"Loading data from {input_path}")
    print(f"Loading embedding model ({args.model_name})...")
    model = SentenceTransformer(args.model_name)

    with open(input_path, 'r') as f:
        data = json.load(f)

    results = []
    print(f"Evaluating {len(data)} samples...")

    for i, item in enumerate(data):
        target = item.get('requested_rewrite', {}).get('target_new', '')
        post = item.get('post', {})
        
        # Extract generation
        fluency = post.get('fluency', {})
        # Note: In our current patched debug script, gen_text is in 'generated_text' or 'fluency'
        gen_list = fluency.get('generated_text', [""])
        gen_text = gen_list[0] if isinstance(gen_list, list) and len(gen_list) > 0 else ""
        
        # If gen_text is empty, look in post directly (fallback for different script versions)
        if not gen_text:
            gen_text_list = post.get('rewrite_gen_content', [""])
            gen_text = gen_text_list[0] if gen_text_list else ""

        if not gen_text or gen_text == "<No Generation>":
            score = 0.0
        else:
            # Compute Cosine Similarity
            try:
                emb1 = model.encode(target, convert_to_tensor=True)
                emb2 = model.encode(gen_text, convert_to_tensor=True)
                score = util.cos_sim(emb1, emb2).item()
            except Exception as e:
                print(f"Error encoding sample {i}: {e}")
                score = 0.0

        # Teacher Accuracy for comparison
        acc_list = post.get('rewrite_acc', [0.0])
        acc = acc_list[0] if isinstance(acc_list, list) and len(acc_list) > 0 else 0.0

        results.append({
            'ID': i,
            'Teacher Acc': round(acc, 4),
            'Embedding Similarity': round(score, 4),
            'Delta': round(score - acc, 4)
        })

    # Save to CSV
    if results:
        keys = results[0].keys()
        with open(output_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

    avg_sim = np.mean([r['Embedding Similarity'] for r in results])
    avg_acc = np.mean([r['Teacher Acc'] for r in results])

    print(f"\nEvaluation Complete!")
    print(f"Average Teacher Accuracy: {avg_acc:.4f}")
    print(f"Average Embedding Similarity: {avg_sim:.4f}")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    evaluate_semantics()

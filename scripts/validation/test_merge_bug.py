import os
import subprocess

def run_test():
    # 1. Path to xu's data
    data_path = 'data/xu_dataset/xu_hallucination_500.json'
    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        return

    # 2. Run the validation script for 20 samples
    cmd = [
        "python3", "scripts/validation/verify_wise_original.py",
        "--data_path", data_path,
        "--max_samples", "20",
        "--output_name", "merge_test_20"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == '__main__':
    run_test()

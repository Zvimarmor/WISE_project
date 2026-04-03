#!/bin/bash
#SBATCH --job-name=wise_merge_test
#SBATCH --output=wise_merge_test_%j.out
#SBATCH --error=wise_merge_test_%j.err
#SBATCH --account=ss-labs
#SBATCH --partition=ss.gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX6000ada"
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zvi.marmor@mail.huji.ac.il


# Define project root
PROJECT_ROOT="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab"

# Environment detection
if [ -f "$HOME/wise_venv/bin/python3" ]; then
    PYTHON="$HOME/wise_venv/bin/python3"
elif [ -f "$HOME/wise_env/bin/python3" ]; then
    PYTHON="$HOME/wise_env/bin/python3"
else
    PYTHON="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab/wise_env/bin/python"
fi

cd "$PROJECT_ROOT"

echo "=========================================================="
echo "LAUNCHING 20-STORY MERGE FREQUENCY TEST"
echo "Targeting merge_freq: 20 (as set in gpt-j-6B.yaml)"
echo "=========================================================="

# Run the validation script for 20 samples
$PYTHON scripts/validation/verify_wise_original.py --add_eos \
    --data_path="data/Xu's_data/xu_dataset_1000_wise.json" \
    --max_samples=20 \
    --output_name="merge_test_results_20"

echo "========== MERGE TEST FINISHED =========="

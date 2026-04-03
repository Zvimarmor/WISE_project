#!/bin/bash
#SBATCH --account=ss-labs
#SBATCH --job-name=xu_1k_eos
#SBATCH --partition=ss.gpu
#SBATCH --output=xu_1k_eos_%j.out
#SBATCH --error=xu_1k_eos_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX6000ada"
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zvi.marmor@mail.huji.ac.il

PROJECT_ROOT="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab"

if [ -f "$HOME/wise_venv/bin/python3" ]; then
    PYTHON="$HOME/wise_venv/bin/python3"
elif [ -f "$HOME/wise_env/bin/python3" ]; then
    PYTHON="$HOME/wise_env/bin/python3"
else
    PYTHON="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab/wise_env/bin/python"
fi

echo "=========================================================="
echo "Starting WISE EOS Token Validation on Xu's 1000 Dataset"
echo "Host: $(hostname)"
echo "Using Python: $PYTHON"
echo "=========================================================="

cd "$PROJECT_ROOT"

# Ensure output directory exists before running
mkdir -p results

# We wrap the data path in quotes because it contains an apostrophe "Xu's_data"
$PYTHON scripts/validation/verify_wise_original.py \
    --data_path="data/Xu's_data/xu_dataset_1000_wise.json" \
    --output_name="xu_dataset_1000_eos_results" \
    --max_samples=1000 \
     \
    --add_hint

echo "========== FINISHED =========="

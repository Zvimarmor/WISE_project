#!/bin/bash
#SBATCH --job-name=llama3.1_wise
#SBATCH --output=llama3.1_%j.out
#SBATCH --error=llama3.1_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=ss.gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zvi.marmor@mail.huji.ac.il

# Define project root
PROJECT_ROOT="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab"

# Robust Python Environment detection
if [ -f "$HOME/wise_venv/bin/python3" ]; then
    PYTHON="$HOME/wise_venv/bin/python3"
elif [ -f "$HOME/wise_env/bin/python3" ]; then
    PYTHON="$HOME/wise_env/bin/python3"
else
    PYTHON="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab/wise_env/bin/python"
fi

echo "=========================================================="
echo "LAUNCHING LLAMA-3.1-8B-INSTRUCT SMOKE TEST (40 SAMPLES)"
echo "Host: $(hostname)"
echo "Using Python: $PYTHON"
echo "=========================================================="

cd "$PROJECT_ROOT"

# Run the Llama-specific validation
$PYTHON scripts/validation/verify_wise_llama.py \
    --data_path="data/Xu's_data/xu_dataset_1000_wise.json" \
    --max_samples=20 \
    --output_name="llama3.1_xu_20"

$PYTHON scripts/validation/verify_wise_llama.py \
    --data_path="data/custom_temporal/extrapolation_1k_dataset.json" \
    --max_samples=20 \
    --output_name="llama3.1_extrap_20"

echo "========== LLAMA-3.1 RUN FINISHED =========="

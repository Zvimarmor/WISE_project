#!/bin/bash
#SBATCH --job-name=extrap_500
#SBATCH --output=extrap_500_%j.out
#SBATCH --error=extrap_500_%j.err
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
echo "LAUNCHING 500-SAMPLE OPTIMIZED EXTRAPOLATION RUN"
echo "Host: $(hostname)"
echo "Using Python: $PYTHON"
echo "=========================================================="

cd "$PROJECT_ROOT"

# Run the optimized validation
# 500 samples from the Extrapolation dataset
$PYTHON scripts/validation/verify_wise_original.py --add_eos \
    --data_path="data/custom_temporal/extrapolation_1k_dataset.json" \
    --max_samples=500 \
    --output_name="extrapolation_500_optimized"

echo "========== 500-SAMPLE RUN FINISHED =========="

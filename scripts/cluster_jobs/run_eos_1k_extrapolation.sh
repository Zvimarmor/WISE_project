#!/bin/bash
#SBATCH --account=ss-labs
#SBATCH --job-name=extrap_1k
#SBATCH --partition=ss.gpu
#SBATCH --output=extrap_1k_%j.out
#SBATCH --error=extrap_1k_%j.err
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
echo "Starting Non-Numerical 1000-Story Extrapolation Test"
echo "Host: $(hostname)"
echo "Using Python: $PYTHON"
echo "=========================================================="

cd "$PROJECT_ROOT"

mkdir -p results

$PYTHON scripts/validation/verify_wise_original.py \
    --data_path="data/custom_temporal/extrapolation_1k_dataset.json" \
    --output_name="extrapolation_1k_eos_results" \
    --max_samples=1000 \
     \
    --add_hint

echo "========== FINISHED =========="

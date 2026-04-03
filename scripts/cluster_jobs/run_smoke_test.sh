#!/bin/bash
#SBATCH --account=ss-labs
#SBATCH --job-name=smoke_test
#SBATCH --partition=ss.gpu
#SBATCH --output=smoke_test_%j.out
#SBATCH --error=smoke_test_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX6000ada"
#SBATCH --time=03:00:00
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
echo "SMOKE TEST: 60 Stories with All Generation Fixes"
echo "Host: $(hostname)"
echo "Using Python: $PYTHON"
echo "=========================================================="

cd "$PROJECT_ROOT"
mkdir -p results
export DEBUG_WISE_TOKENS=1

# Run on Extrapolation dataset (20 stories)
echo "--- Extrapolation Dataset (20 stories) ---"
$PYTHON scripts/validation/verify_wise_original.py --add_eos \
    --data_path="data/custom_temporal/extrapolation_1k_dataset.json" \
    --output_name="smoke_test_extrap_20" \
    --max_samples=20 \
    

echo "--- Xu Dataset (20 stories) ---"
$PYTHON scripts/validation/verify_wise_original.py --add_eos \
    --data_path="data/Xu's_data/xu_dataset_1000_wise.json" \
    --output_name="smoke_test_xu_20" \
    --max_samples=20 \
    

echo "========== SMOKE TEST FINISHED =========="

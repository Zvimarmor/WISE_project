#!/bin/bash
#SBATCH --job-name=llama_smoke_60
#SBATCH --output=llama_smoke_%j.out
#SBATCH --error=llama_smoke_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=haim.gpu
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
echo "LAUNCHING 60-STORY LLAMA-3.1-8B-INSTRUCT SMOKE TEST"
echo "Host: $(hostname)"
echo "Using Python: $PYTHON"
echo "=========================================================="

cd "$PROJECT_ROOT"

# Run the Llama validation
# 60 samples from the Xu dataset
$PYTHON scripts/validation/verify_wise_llama.py \
    --data_path="data/Xu's_data/xu_dataset_1000_wise.json" \
    --max_samples=60 \
    --output_name="llama_smoke_60_results" \
    --add_eos

echo "========== LLAMA SMOKE TEST FINISHED =========="

#!/bin/bash
#SBATCH --account=ss-labs
#SBATCH --job-name=wise_custom
#SBATCH --partition=ss.gpu
#SBATCH --output=wise_custom_%j.out
#SBATCH --error=wise_custom_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX6000ada"
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zvi.marmor@mail.huji.ac.il

# 1. Environment Setup
PROJECT_ROOT="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab"

if [ -f "$HOME/wise_venv/bin/python3" ]; then
    VENV_PYTHON="$HOME/wise_venv/bin/python3"
elif [ -f "$HOME/venv/bin/python3" ]; then
    VENV_PYTHON="$HOME/venv/bin/python3"
else
    echo "No valid Virtual Environment found in Home!"
    exit 1
fi

echo "Using Python from: $VENV_PYTHON"
set -e

# 2. Execution
export PYTHONPATH=$PROJECT_ROOT/EasyEdit:$PYTHONPATH
cd $PROJECT_ROOT

echo "Starting Custom Temporal OOD validation (2023/2024)..."
$VENV_PYTHON scripts/validation/verify_wise_original.py \
    --data_path data/custom_temporal_2023.json \
    --output_name custom_temporal_2023_results

echo "Verification complete."

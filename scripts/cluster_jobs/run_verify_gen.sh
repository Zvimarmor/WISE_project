#!/bin/bash
#SBATCH --account=ss-labs
#SBATCH --job-name=wise_verify_gen
#SBATCH --partition=ss.gpu
#SBATCH --output=verify_gen_%j.out
#SBATCH --error=verify_gen_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX6000ada"
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zvi.marmor@mail.huji.ac.il

# 1. Environment Setup
PROJECT_ROOT="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab"

# Check Home directory venv first (safest for permissions)
if [ -f "$HOME/wise_venv/bin/python3" ]; then
    VENV_PYTHON="$HOME/wise_venv/bin/python3"
elif [ -f "$HOME/venv/bin/python3" ]; then
    VENV_PYTHON="$HOME/venv/bin/python3"
elif [ -f "$PROJECT_ROOT/venv/bin/python3" ]; then
    VENV_PYTHON="$PROJECT_ROOT/venv/bin/python3"
else
    echo "No valid Virtual Environment found in Home or Project Root!"
    exit 1
fi

echo "Using Python from: $VENV_PYTHON"

export PYTHONPATH=$PROJECT_ROOT/EasyEdit:$PYTHONPATH
cd $PROJECT_ROOT

# Run verification script using the absolute local venv python path
$VENV_PYTHON scripts/verify_generation.py

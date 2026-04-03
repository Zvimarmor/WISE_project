#!/bin/bash
#SBATCH --account=ss-labs
#SBATCH --job-name=wise_ood_strict
#SBATCH --partition=ss.gpu
#SBATCH --output=wise_ood_%j.out
#SBATCH --error=wise_ood_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX6000ada"
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zvi.marmor@mail.huji.ac.il

# 1. Environment Setup
PROJECT_ROOT="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab"

# Check Home directory venv
if [ -f "$HOME/wise_venv/bin/python3" ]; then
    VENV_PYTHON="$HOME/wise_venv/bin/python3"
    VENV_PIP="$HOME/wise_venv/bin/pip"
elif [ -f "$HOME/venv/bin/python3" ]; then
    VENV_PYTHON="$HOME/venv/bin/python3"
    VENV_PIP="$HOME/venv/bin/pip"
else
    echo "No valid Virtual Environment found in Home!"
    exit 1
fi

echo "Using Python from: $VENV_PYTHON"

set -e

# 2. Execution
export PYTHONPATH=$PROJECT_ROOT/EasyEdit:$PYTHONPATH
cd $PROJECT_ROOT

# RUN 1: Strict Temporal OOD (14 Stories)
echo "Starting Strict 2021+ Temporal OOD validation..."
$VENV_PYTHON scripts/validation/verify_wise_original.py \
    --data_path data/temporal_validation_filtered.json \
    --output_name temporal_strict_19.3.26

# RUN 2: Strict Wikibio Hallucination (306 Stories)
# echo "Starting Strict Wikibio Hallucination validation..."
# $VENV_PYTHON scripts/validation/verify_wise_original.py \
#     --data_path data/wikibio-test-filtered.json \
#     --output_name wikibio_strict_19.3.26

echo "Verification suite complete."

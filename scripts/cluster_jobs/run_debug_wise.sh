#!/bin/bash
#SBATCH --account=ss-labs
#SBATCH --mem=40g
#SBATCH --time=08:00:00
#SBATCH --job-name=WISE_Rand250
#SBATCH --error=debug_wise_rand_%j.err
#SBATCH --output=debug_wise_rand_%j.out
#SBATCH --partition=gpu.q
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100"
#SBATCH --cpus-per-task=4

# 1. Environment Setup
module load python/3.11.13

PROJECT_ROOT="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab_Project"

if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "Virtual environment not found!"
    exit 1
fi

# 2. Run Debug Script
cd $PROJECT_ROOT
echo "Starting Debug Run..."
python3 scripts/debug_wise_metric.py
echo "Done."

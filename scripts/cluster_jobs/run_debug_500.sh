#!/bin/bash
#SBATCH --account=ss-labs
#SBATCH --mem=48g
#SBATCH --time=24:00:00
#SBATCH --job-name=WISE_Rand500
#SBATCH --error=debug_wise_rand500_%j.err
#SBATCH --output=debug_wise_rand500_%j.out
#SBATCH --partition=ss.gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX6000ada"
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

# 2. Run Debug Script with 500 samples
cd $PROJECT_ROOT
echo "Starting Debug Run (500 samples)..."
python3 scripts/debug_wise_metric.py --num_samples 500
echo "Done."

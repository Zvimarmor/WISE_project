#!/bin/bash
#SBATCH --account=ss-labs
#SBATCH --mem=48g
#SBATCH --time=48:00:00
#SBATCH --job-name=WISE_Full
#SBATCH --error=debug_wise_full_%j.err
#SBATCH --output=debug_wise_full_%j.out
#SBATCH --partition=ss.gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX6000ada"
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zvimarmor@gmail.com

# 1. Environment Setup
module load python/3.11.13

# Using absolute path to avoid Slurm spooling issues with relative detection
PROJECT_ROOT="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab"
WORKING_PROJECT="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab_Project"
echo "Project Root: $PROJECT_ROOT"

if [ -f "$WORKING_PROJECT/venv/bin/activate" ]; then
    source "$WORKING_PROJECT/venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "Virtual environment not found in either $PROJECT_ROOT or $WORKING_PROJECT!"
    exit 1
fi

# 2. Run Debug Script with 0 (All) samples
cd $PROJECT_ROOT
echo "Starting Debug Run (Full Dataset)..."
python3 scripts/debug_wise_metric.py --num_samples 0
echo "Done."

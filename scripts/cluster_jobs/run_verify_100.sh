#!/bin/bash
#SBATCH --account=ss-labs
#SBATCH --job-name=wise_verify_100
#SBATCH --partition=ss.gpu
#SBATCH --output=wise_verify_100_%j.out
#SBATCH --error=wise_verify_100_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX6000ada"
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zvi.marmor@mail.huji.ac.il

# 1. Environment Setup
module load python/3.11.13

PROJECT_ROOT="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab"
WORKING_PROJECT="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab_Project"

if [ -f "$WORKING_PROJECT/venv/bin/activate" ]; then
    source "$WORKING_PROJECT/venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "Virtual environment not found!"
    exit 1
fi

export PYTHONPATH=$PROJECT_ROOT/EasyEdit:$PYTHONPATH
cd $PROJECT_ROOT

# Run verification script
python3 scripts/verify_fixes_100.py --num_samples 100 --eval_steps 10

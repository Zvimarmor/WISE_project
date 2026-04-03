#!/bin/bash
#SBATCH --account=ss-labs
#SBATCH --job-name=wise_wiki_200
#SBATCH --partition=ss.gpu
#SBATCH --output=wiki_200_%j.out
#SBATCH --error=wiki_200_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX6000ada"
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zvi.marmor@mail.huji.ac.il

# 1. Environment Setup
PROJECT_ROOT="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab"

# Check Home directory venv first
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
# 2. Dependency Check (Ensure ALL dependencies are installed)
echo "Checking/Installing all dependencies from requirements.txt..."
$VENV_PIP install -r $PROJECT_ROOT/EasyEdit/requirements.txt
$VENV_PIP install rouge-score sentence-transformers 

# 3. Execution
export PYTHONPATH=$PROJECT_ROOT/EasyEdit:$PYTHONPATH
cd $PROJECT_ROOT

echo "Starting 200-story Temporal validation run..."
$VENV_PYTHON scripts/validation/verify_wise_original.py \
    --data_path data/temporal/temporal-edit.json \
    --output_name temporal_validation_200

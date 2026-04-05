#!/bin/bash
#SBATCH --job-name=gptj_30
#SBATCH --output=gptj_30_%j.out
#SBATCH --error=gptj_30_%j.err
#SBATCH --account=ss-labs
#SBATCH --partition=ss.gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --exclude=elscn-[60-64]
#SBATCH --time=04:00:00
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
echo "Job Run on node: $(hostname)"
echo "Starting Lifelong Sequential Editing Memory Test (GPT-J 6B)"
echo "=========================================================="

cd "$PROJECT_ROOT"

# Pre-download strictly required NLTK resources
$PYTHON -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Execution command for the 30-story sequential memory test
$PYTHON scripts/validation/verify_wise_original.py \
    --hparams_dir EasyEdit/hparams/WISE/gpt-j-6B.yaml \
    --data_path data/temporal/custom_temporal_2023.json \
    --num_samples 30 \
    --results_folder results/gptj_sequential_retention_30 \
    --add_eos

echo "Job finished at $(date)"

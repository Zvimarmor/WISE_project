#!/bin/bash
#SBATCH --account=ss-labs
#SBATCH --job-name=wise_eos
#SBATCH --partition=ss.gpu
#SBATCH --output=wise_eos_%j.out
#SBATCH --error=wise_eos_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX6000ada"
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zvi.marmor@mail.huji.ac.il

PROJECT_ROOT="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab"

if [ -f "$HOME/wise_venv/bin/python3" ]; then
    PYTHON="$HOME/wise_venv/bin/python3"
elif [ -f "$HOME/wise_env/bin/python3" ]; then
    PYTHON="$HOME/wise_env/bin/python3"
else
    PYTHON="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab/wise_env/bin/python"
fi

echo "=========================================================="
echo "Starting WISE EOS Token Suppression Test on $(hostname)"
echo "Using Python: $PYTHON"
echo "=========================================================="

cd $PROJECT_ROOT

$PYTHON scripts/validation/verify_wise_original.py \
    --data_path=data/custom_temporal_2023.json \
    --output_name=custom_temporal_eos_test \
    

echo "========== FINISHED =========="

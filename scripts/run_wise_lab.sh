#!/bin/bash
#SBATCH --account=ss-labs
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#SBATCH --job-name=WISE_Lab_Data_Run
#SBATCH --error=/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab_Project/logs/wise_lab_%A_%a.err
#SBATCH --output=/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab_Project/logs/wise_lab_%A_%a.out
#SBATCH --partition=ss.gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX6000ada"
#SBATCH --cpus-per-task=8

# 1. Environment Setup
module load python/3.11.13

PROJECT_ROOT="/ems/elsc-labs/sompolinsky-h/zvi.marmor/WISE_Lab_Project"

if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "Warning: venv not found at $PROJECT_ROOT/venv. Please create it."
    exit 1
fi

export PYTHONPATH=$PROJECT_ROOT/EasyEdit:$PYTHONPATH

cd $PROJECT_ROOT
echo "Changed directory to: $(pwd)"

echo "Starting WISE Run on Lab Data (Full Run)"

# 2. Configuration
# Choose which dataset to run: 'text', 'qa', or 'zsre'
# You can change this manually or pass it as an argument ($1)
# NOTE: 'qa' dataset now contains rule-based rephrases for generalization testing
MODE="${1:-text}"    # Default: Text Editing for this run
#MODE="${1:-qa}"    # Default: QA fact editing with rephrased questions
TRACK_METRIC="${2:-true}"

REPO_ROOT=$PROJECT_ROOT
DATA_ROOT="$REPO_ROOT/data/lab_wise"
HPARAMS_DIR="$REPO_ROOT/EasyEdit/hparams/WISE/gpt-j-6B.yaml"

# 3. Dynamic Data Setup (Handling WISE Filenames)
TEMP_DATA_DIR="$REPO_ROOT/data/temp_run_${SLURM_JOB_ID}"

if [ "$MODE" == "text" ]; then
    echo "Mode: Text Paragraph Editing"
    DATA_TYPE="temporal"
    mkdir -p "$TEMP_DATA_DIR/temporal"
    # Link Lab files to expected WISE names
    ln -s "$DATA_ROOT/temporal/lab_wise_text_edit.json" "$TEMP_DATA_DIR/temporal/temporal-edit.json"
    ln -s "$DATA_ROOT/temporal/lab_wise_text_train.json" "$TEMP_DATA_DIR/temporal/temporal-train.json"

elif [ "$MODE" == "qa" ]; then
    echo "Mode: QA Fact Editing"
    DATA_TYPE="temporal"
    mkdir -p "$TEMP_DATA_DIR/temporal"
    ln -s "$DATA_ROOT/temporal/lab_wise_qa_edit.json" "$TEMP_DATA_DIR/temporal/temporal-edit.json"
    ln -s "$DATA_ROOT/temporal/lab_wise_qa_train.json" "$TEMP_DATA_DIR/temporal/temporal-train.json"

elif [ "$MODE" == "zsre" ]; then
    echo "Mode: ZsRE Format"
    DATA_TYPE="ZsRE"
    TEMP_DATA_DIR="$REPO_ROOT/data/lab_zsre" 
else
    echo "Error: Unknown MODE '$MODE'. Use 'text', 'qa', or 'zsre'."
    exit 1
fi

echo "Data prepared at: $TEMP_DATA_DIR"

# Build arguments
# Using generate-text to avoid OOM and allow offline analysis
ARGS="--editing_method WISE \
    --hparams_dir $HPARAMS_DIR \
    --data_dir $TEMP_DATA_DIR \
    --data_type $DATA_TYPE \
    --sequential_edit \
    --ds_size 3000 \
    --output_dir $REPO_ROOT/outputs/wise_${MODE}_results \
    --evaluation_type generate-text"

if [ "$TRACK_METRIC" == "true" ]; then
    ARGS="$ARGS --track_first_edit --eval_steps 10"
    echo "Tracking First Edit Retention enabled (every 10 steps)."
fi

# 4. Run WISE
python EasyEdit/examples/run_wise_editing.py $ARGS

# 5. Cleanup
if [ "$MODE" != "zsre" ]; then
    echo "Cleaning up temp data..."
    rm -rf "$TEMP_DATA_DIR"
fi

echo "Done. Results saved to outputs/wise_${MODE}_results"

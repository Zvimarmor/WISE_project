# Lab Datasets & Cluster Execution Guide

This document details the new datasets created from the Lab's wiki data and provides a complete guide for running the WISE model on the Lab's cluster.

## 1. New Lab Datasets Analysis

We created three distinct dataset variations from `filtered_wiki_dataset_with_knowledge_instruct_facts.json`.

### A. Lab-Text Dataset (`lab_wise_text`)
*   **Location**: `data/lab_wise/lab_wise_text_edit.json`
*   **Format**: Matches WISE `temporal` data type.
*   **Goal**: Teach the model full paragraphs of new knowledge.
*   **Field Mapping**:
    *   `prompt`: `"Tell me about [Title]"`
    *   `target_new`: The full `text` paragraph (e.g., "Mostafa Asal is a...").
    *   `ood_rephrase`: A paraphrase from the source (e.g., "Summary of Mostafa Asal...").
    *   `locality_prompt`: A random question from a *different* article (to ensure specific editing).
*   **Comparison**: This is the closest equivalent to the standard `temporal` dataset used in the WISE paper, but with richer, cleaner 2024 data.

### B. Lab-QA Dataset (`lab_wise_qa`)
*   **Location**: `data/lab_wise/lab_wise_qa_edit.json`
*   **Format**: Matches WISE `temporal` data type.
*   **Goal**: Teach the model specific atomic facts via Q&A.
*   **Field Mapping**:
    *   `prompt`: Original Question (`q`).
    *   `target_new`: Original Answer (`a`).
    *   `ood_rephrase`: Synthesized as `"Question: [Q] \n Answer:"`.
    *   `locality_prompt`: A random question from a *different* article.
*   **Comparison**: Structurally similar to `temporal` but content-wise focused on short factoid editing, making it a hybrid between ZsRE and Temporal tasks.

### C. Lab-ZsRE Dataset (`lab_zsre`)
*   **Location**: `data/lab_zsre/ZsRE/zsre_mend_edit.json`
*   **Format**: Matches WISE `ZsRE` data type EXACTLY.
*   **Goal**: Evaluate using the standard ZsRE metrics (Exact Match) but on your custom data.
*   **Field Mapping**:
    *   `src`: Question.
    *   `alt`: Answer.
    *   `loc`: Random question from different article.
    *   `cond`: `Title >> Answer || Question`.
*   **Comparison**: Identical format to the standard ZsRE benchmark, allowing direct use of the `ZsRE` dataloader in WISE.

---

## 2. Cluster Execution Plan

To run on the cluster, we need to adapt the provided SLURM script. 

### Critical: File naming for WISE
The `run_wise_editing.py` script **hardcodes** the input filenames.
*   If you use `--data_type temporal`, it looks for `temporal-edit.json`.
*   If you use `--data_type ZsRE`, it looks for `zsre_mend_edit.json`.

**Solution**: The SLURM script below creates minimal temporary directories with symbolic links named correctly. This allows us to run `Lab-Text` and `Lab-QA` without modifying the python code.

### Prerequisites
1.  **Environment**: Ensure your `venv` has `easyeditor` installed (or `requirements.txt` from this repo).
2.  **GPU**: WISE requires significant VRAM. The script requests `gpu:1` (likely A100/A6000 on your cluster). If you get OOM (Out Of Memory) errors, you may need to reduce batch size to 1 (which is default for WISE anyway).

### The SLURM Script (`scripts/run_wise_lab.sh`)

Save this file to your `scripts/` folder. It runs the **Lab-Text** dataset by default.

```bash
#!/bin/bash
#SBATCH --account=ss-labs
#SBATCH --mem=64g
#SBATCH --time=4:00:00
#SBATCH --job-name=WISE_Lab_Data_Run
#SBATCH --error=./logs/wise_lab_%A_%a.err
#SBATCH --output=./logs/wise_lab_%A_%a.out
#SBATCH --partition=gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# 1. Environment Setup
# Replace this with YOUR actual environment path if different from the example
source /ems/elsc-labs/sompolinsky-h/yoni.ankri/Repos/CNNtoRNN/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

echo "Starting WISE Run on Lab Data"

# 2. Configuration
# Choose which dataset to run: 'text', 'qa', or 'zsre'
MODE="text" 

# Base paths
REPO_ROOT=$(pwd) # Assuming you submit from the repo root
DATA_ROOT="$REPO_ROOT/data/lab_wise"
HPARAMS_DIR="$REPO_ROOT/EasyEdit/hparams/WISE/gpt-j-6B.yaml"

# 3. Dynamic Data Setup (Handling WISE Filenames)
# WISE script expects specific folders/filenames. We construct them on the fly.
TEMP_DATA_DIR="$REPO_ROOT/data/temp_run_${SLURM_JOB_ID}"

if [ "$MODE" == "text" ]; then
    echo "Mode: Text Paragraph Editing"
    DATA_TYPE="temporal"
    mkdir -p "$TEMP_DATA_DIR/temporal"
    # Link Lab files to expected WISE names
    ln -s "$DATA_ROOT/lab_wise_text_edit.json" "$TEMP_DATA_DIR/temporal/temporal-edit.json"
    ln -s "$DATA_ROOT/lab_wise_text_train.json" "$TEMP_DATA_DIR/temporal/temporal-train.json"

elif [ "$MODE" == "qa" ]; then
    echo "Mode: QA Fact Editing"
    DATA_TYPE="temporal"
    mkdir -p "$TEMP_DATA_DIR/temporal"
    ln -s "$DATA_ROOT/lab_wise_qa_edit.json" "$TEMP_DATA_DIR/temporal/temporal-edit.json"
    ln -s "$DATA_ROOT/lab_wise_qa_train.json" "$TEMP_DATA_DIR/temporal/temporal-train.json"

elif [ "$MODE" == "zsre" ]; then
    echo "Mode: ZsRE Format"
    DATA_TYPE="ZsRE"
    TEMP_DATA_DIR="$REPO_ROOT/data/lab_zsre" # This one is already formatted correctly
fi

echo "Data prepared at: $TEMP_DATA_DIR"

# 4. Run WISE
# Note: Sequential edit is ON for lifelong learning simulation
python EasyEdit/examples/run_wise_editing.py \
    --editing_method WISE \
    --hparams_dir "$HPARAMS_DIR" \
    --data_dir "$TEMP_DATA_DIR" \
    --data_type "$DATA_TYPE" \
    --sequential_edit \
    --ds_size 3000 \
    --output_dir "$REPO_ROOT/outputs/wise_${MODE}_results"

# 5. Cleanup
if [ "$MODE" != "zsre" ]; then
    rm -rf "$TEMP_DATA_DIR"
fi

echo "Done. Results saved to outputs/wise_${MODE}_results"
```

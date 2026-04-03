# WISE-Llama: Scaling Model Editing to 3.1-Instruct

This repository contains the code and configuration for scaling and validating the **WISE (Writing Into Subspaces)** model editing framework on **Llama-3.1-8B-Instruct**.

## Project Overview
The goal of this project is to evaluate the sequential editing performance of the WISE algorithm. We have transitioned from the original GPT-J implementation to support Llama-3.1-Instruct with optimizations for memory efficiency (fp16) and generation stability (Chat Templates).

## Key Components

### 1. Model Editing (EasyEdit)
We use a modified version of the `EasyEdit` library tailored for WISE:
- `EasyEdit/easyeditor/models/wise/`: Core WISE implementation (TIES merge, Gradient Masking).
- `EasyEdit/hparams/WISE/`: Critical hyperparameter files.
  - `llama-3.1-8b-instruct.yaml`: Configured for **edit_lr: 0.1** and **use_chat_template: true**.

### 2. Validation Scripts (`scripts/validation/`)
- `verify_wise_llama.py`: The main entry point for Llama evaluation.
  - Implements **Llama Chat Template** wrapping for every prompt.
  - Uses **Batch SBERT Encoding** for high-performance semantic similarity scoring.
  - Handles automatic NLTK dependency downloads for cluster nodes.
- `verify_wise_original.py`: Legacy script for GPT-J/standard model validation.

### 3. Cluster Execution (`scripts/cluster_jobs/`)
SLURM scripts optimized for the `haim.gpu` partition (High-VRAM RTX Pro/A100 GPUs):
- `run_wise_llama_smoke_60.sh`: Quick 60-story validation to verify parameters.
- `run_wise_extrap_500.sh`: Full scale 500-sample extrapolation run.
- `run_wise_xu_500.sh`: Baseline run for the Xu dataset.

## How to Use

### Setup
1. Create a virtual environment and install dependencies from `requirements.txt`.
2. Set up your Hugging Face token at `~/.cache/huggingface/token`.
3. For cluster runs, ensure the path in the `.sh` files reflects your local environment.

### Running a Smoke Test
To verify the setup on a small sample (60 stories) with Llama:
```bash
sbatch scripts/cluster_jobs/run_wise_llama_smoke_60.sh
```

### Analysis & Reporting
Results are saved to `results/llama_instruct_test/` as JSON. 
We have included various analysis tools in `scripts/analysis/` to generate Markdown reports and plot loss curves.

## Recent Optimizations
- **Learning Rate Tuning**: Lowered `edit_lr` to `0.1` to prevent "babbling" and preserve base model language capabilities.
- **Instruct Integration**: Forced Chat Template usage to ensure the model follows context instead of hallucinating.
- **Stop Criteria**: Added custom stop markers to handle Llama-specific end-of-turn tokens effectively.

---
*Developed as part of the Sompolinski Lab WISE scaling project.*

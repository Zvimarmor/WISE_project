# Project Journal: WISE Model Editing Implementation

## Overview
**Goal**: Reproduce and experiment with the methodology from the paper *"WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models"*.
**Objective**: Implement a lifelong model editing system that maintains Reliability, Generalization, and Locality while minimizing side effects (Hallucination, Temporal OOD issues).

---

## Step 1: Data Collection & Validation
**Date**: 2026-01-12
**Status**: COMPLETE

### What We Did
1.  **Investigated Requirements**: Analyzed the WISE paper and `EasyEdit` code to identify the three required datasets:
    *   **ZsRE** (Reliability/Generalization/Locality).
    *   **SelfCheckGPT** (Hallucination).
    *   **Temporal** (Out-of-Distribution Generalization).
2.  **Acquired Data**:
    *   **ZsRE**: Downloaded from HuggingFace (`wangzn2001/data`). Verified 19,086 edit samples.
    *   **Hallucination & Temporal**: Located the raw source data in the `zjunlp/KnowEdit` repository (same authors).
        *   Hallucination = `WikiBio` benchmark (770 total samples).
        *   Temporal = `WikiRecent` benchmark (1,836 total samples).
3.  **Verified Integrity**:
    *   Checked file schemas against WISE code requirements.
    *   Verified sample counts match or exceed paper benchmarks.
    *   Confirmed Locality data (Natural Questions, RedPajama) is embedded within the datasets.

### Key Outputs
*   **Data Store**: Full datasets available in `data/`.
*   **Documentation**: `docs/datasets_summary.md` details the schema and provides 10 examples for each dataset.
*   **Ready for**: Step 2 (Experimentation).

## Step 2: Temporal Data Correction & Lab Data Analysis
**Date**: 2026-01-20
**Status**: COMPLETE

### What We Did
1.  **Corrected Temporal Dataset**:
    *   Identified that the initial "Temporal" dataset (KnowEdit) was incorrect for OOD evaluation.
    *   Located and downloaded the correct **"Canonical Examples" dataset (Hewitt et al.)**.
    *   Created `temporal-edit.json` (Edit) and `temporal-train.json` (Locality), verifying the critical "Self-driving cars" OOD paragraph exists.
2.  **Updated Documentation**:
    *   Revised `docs/wise_technical_summary.md` to accurately reflect the OOD Generalization experiments using the new dataset.
3.  **Lab Data Analysis**:
    *   Analyzed `progress_phase_50_year_2014` datasets for age distribution.
    *   Generated plots (`plots/lab's_data_plot1.png`, `...plot2.png`) showing story counts per age.
    *   Investigated `filtered_wiki_dataset_with_knowledge_instruct_facts.json`, confirming it contains recent 2024 data and is likely **not** the source for the 1965-2015 bio generation, explaining the data gap for years 1973/1987.

## Step 3: Lab Dataset Creation & Cluster Prep
**Date**: 2026-02-09
**Status**: COMPLETE

### What We Did
1.  **Created Lab Datasets**:
    *   Processed `filtered_wiki_dataset_with_knowledge_instruct_facts.json` (2024 Lab Data).
    *   Wrote `data/lab_wise/create_datasets.py` to generate three WISE-compatible datasets:
        *   **Lab-Text**: Paragraph editing (closest to original WISE Temporal task).
        *   **Lab-QA**: Specific fact editing (closer to ZsRE/MEND task).
        *   **Lab-ZsRE**: A direct clone of the ZsRE structure for benchmarking.
2.  **Documentation**:
    *   Created `docs/lab_datasets_and_cluster_guide.md` detailing the exact schema mapping and purpose of each new dataset.
3.  **Cluster Execution**:
    *   Developed `scripts/run_wise_lab.sh`, a SLURM script adapted for the Lab's cluster, targeting A100/V100S GPUs to avoid OOM.

## Step 4: Full Scale Comparisons (Text vs. QA) & Metric Verification
**Date**: 2026-02-14 to 2026-02-17
**Status**: COMPLETE

### What We Did & Key Results
1.  **Run 1: Text Editing (Paragraphs)**
    *   **Date**: Feb 14, 2026
    *   **Scope**: 986 edits on Lab-Text dataset.
    *   **Results**:
        *   **Reliability**: 83.4%
        *   **Generalization**: **90.6%** (Very high performance on paragraph completion).
        *   **Locality**: 70.7%
    *   **Outcome**: Successfully validated the WISE pipeline on long-form text editing.

2.  **Run 2: QA Editing (Fact-Based)**
    *   **Date**: Feb 16, 2026
    *   **Scope**: 2958 edits on Lab-QA dataset.
    *   **Results**:
        *   **Reliability**: **~100%** (Perfect retention of exact edits).
        *   **Generalization**: **65.2%** (Significantly lower than Text).
    *   **Key Discovery**: Observed a "Generalization Decay." While the model perfectly remembers 2958 facts (Reliability), its ability to answer *rephrased* questions about early edits drops significantly (from ~90% to <20%) as the memory fills up. This suggests retrieval interference for non-exact queries.

3.  **Metric Simulation & Debugging**:
    *   **Goal**: Investigate if low scores (e.g., Generalization 65%) were due to model failure or strict metrics.
    *   **Simulation**: Created `scripts/simulate_wise_evaluation.py` to test the official evaluation code against realistic outputs.
    *   **Finding**: The metric is extremely strict.
        *   **QA**: Trivial rephrasing yields **0.00**.
        *   **Text**: Slight drift yields **~0.20**.
    *   **Conclusion**: Model performance is likely higher than scores suggest.
    *   **Current Action**: Scale up the evaluation to 250 samples with qualitative checking.

## Step 5: Scaling up & Optimization (Toward Final Report)
**Date**: 2026-02-18 to 2026-02-19
**Status**: IN PROGRESS

### What We Did
1.  **Resolved Resource Bottlenecks**:
    *   Identified that **16GB V100s** were causing OOM during sequential editing of 3000 samples.
    *   Reconfigured SLURM scripts (`run_wise_lab.sh`) to strictly target **48GB RTX6000ada** and **80GB A100** nodes via constraints.
2.  **Implemented "Generate & Save" Strategy**:
    *   Modified `run_wise_editing.py` to support `evaluation_type: 'generate-text'`.
    *   Model now saves the raw generated strings to JSON rather than computing slow/strict metrics online.
    *   This allows for offline "BERTScore" or "LLM-Judge" analysis, which is more representative of real-world utility.
3.  **Launched Large-Scale Experiments**:
    *   **Run 3 (Full)**: Processing 3000 Lab-Text samples on `ss.gpu`.
    *   **Run 4 (Randomized 250)**: A new experiment with 250 randomly shuffled samples on `gpu.q` (A100) to get a statistically significant qualitative baseline.
4.  **Stability & Safety**:
    *   Implemented **Incremental Metric Saving** in `editor.py`. Metrics are now saved to `all_metrics_intermediate.json` after *every* edit.
    *   This ensures that even if a job times out or crashes after 1000 edits, no data is lost.

### Next Steps
*   Analyze the 250-sample results (`debug_wise_rand250.json`) to calculate a more accurate "Semantic Success" score.
*   Generate the final Word report for the PI.
*   Establish a permanent GitHub repository for the codebase.

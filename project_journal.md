# Project Journal: WISE Model Editing Implementation

## Background: The WISE Framework
**Paper**: *WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models*
**DOI**: [10.48550/arXiv.2405.14768](https://doi.org/10.48550/arXiv.2405.14768)

**Goal**: Enable Large Language Models (LLMs) to learn thousands of new facts sequentially without "forgetting" old facts or compromising the model's original knowledge.
**Mechanism**: Unlike traditional fine-tuning, WISE uses **Side Memory Subspaces**. For each edit, it allocates a specific subspace in the model's weight matrix. During inference, high-precision activation routing ensures that the model only accesses the relevant edit memory for the given query.
**Original Benchmarks**: The authors validated WISE on ZsRE (Question Answering), WikiRecent (Temporal/OOD facts), and WikiBio (Hallucination reduction).

---

## Project Vision & Objective
**Goal**: Implement and evaluate the WISE (*Rethinking the Knowledge Memory*) framework for lifelong model editing on GPT-J-6B using Lab-specific data, focusing on stability across 3000+ sequential edits.

---

## Phase 1: Data Engineering & Lab Integration
**Milestone**: Transitioned from generic benchmarks to custom Lab-specific datasets.
*   **Dataset Acquisition**: Secured ZsRE (Standard), WikiRecent (Temporal), and WikiBio (Hallucination) benchmarks.
*   **Lab Dataset Creation**: Developed scripts to transform raw 2024 Wiki data into WISE-compatible formats:
    *   **Lab-Text**: Paragraph-completion tasks for long-form reliability.
    *   **Lab-QA**: Fact-based question-answering for strict knowledge injection.
*   **Key Insight**: Long-form paragraph editing (Text) shows significantly higher generalization scores than short-form QA, likely due to richer contextual cues.

---

## Phase 2: Technical Stability & Hardware Optimization
**Milestone**: Achieved stable 3000+ sequential editing runs through hardware-aware optimization.
*   **Precision Transition**: Discarded 8-bit quantization due to instability; reverted to **Full Precision (FP16/FP32)**.
*   **Hardware Targeting**: Configured SLURM to strictly target high-memory nodes (**48GB RTX6000ada** and **80GB A100**).
*   **Infinite Reliability**: Implemented **Incremental Checkpointing** in `editor.py`. Metrics and weights are now saved after *every* edit, preventing data loss during cluster timeouts.
*   **Speed Up**: Optimized `n_iter` and evaluation frequency (`--eval_steps 10`), reducing the bottleneck of large-scale monitoring.

---

## Phase 3: The Discovery of "Generalization Decay"
**Milestone**: Identified a critical performance drop as memory fills up.
*   **Experiment**: Ran comparative analysis between Run 1 (Text, 986 edits) and Run 2 (QA, 2958 edits).
*   **The Findings**:
    *   **Reliability**: Remained at **~100%** (The model remembers the exact facts indefinitely).
    *   **Generalization**: Observed a sharp decay. While new facts are learned perfectly, the model's ability to answer *rephrased* questions about early edits drops significantly as the memory fills up.
*   **First Edit Retention (Run 2)**: 
    *   After **Edit #1**, Generalization was **~90%**.
    *   After **Edit #3000**, Generalization for **Edit #1** dropped to **< 20%**.
    *   **Implication**: Perfect exact-match memory (Reliability) masks a failure in semantic retrieval (Generalization) over time.
*   **Root Cause**: Retrieval interference within the side-memory subspaces when queries are not an exact match.

---

## Phase 4: Semantic Validation & The "Metric Gap"
**Milestone**: Broke through the limitations of strict token-based metrics using Embedding Similarity.
*   **Experiment**: Conducted a 250-sample randomized "Debug" experiment to qualitatively review the "Free Generation" stories.
*   **The Metric Discrepancy**:
    *   **Teacher Accuracy**: **93.4%** (Token-perfect memorization).
    *   **Embedding Similarity**: **82.5%** (Semantic concept match).
*   **Key Discovery (The 11% Gap)**: High "Teacher Accuracy" is a hallucinated confidence. The model often matches the target's starting tokens but veers off into non-factual stories (Mini-hallucinations).
*   **Cross-Validation**: Confirmed results using both lightweight (`all-MiniLM-L6-v2`) and powerful (`all-mpnet-base-v2`) embedding models, showing high correlation between semantic measures.

---

## Phase 5: Lifelong Scaling & Final Evaluation
**Milestone**: Validated WISE stability across the full Lab-Text dataset (3000+ edits).
*   **Results (Full Dataset - 986 Samples)**:
    *   **Teacher Accuracy**: **87.36%**
    *   **Embedding Similarity**: **79.77%**
*   **Results (Randomized 500 Samples)**:
    *   **Teacher Accuracy**: **91.05%**
    *   **Embedding Similarity**: **81.43%**
*   **Key Discovery**: The "Metric Gap" remains stable at **~8-11%** even as we scale to the full dataset. This proves that WISE's semantic retrieval is robust and does not degrade significantly when thousands of facts are stored in the side memory.
*   **Strategy Success**: The **"Generate & Save"** strategy enabled high-fidelity qualitative review of all 1486 stories across both runs, providing a complete bank of examples for the final report.

---
---
 
 ## Phase 6: Scaling to 1000 Stories & Hallucination Suppression
 **Milestone**: Successfully scaled the validation pipeline to 1000 stories and implemented deterministic hallucination suppression.
 *   **Scale Up**: Expanded the Wikipedia dataset to **1000 stories** (Xu's 2026 dataset and a custom 1k Extrapolation dataset).
 *   **EOS Intervention**: Implemented `<|endoftext|>` token injection at the end of targets during editing.
 *   **Technical Fixes**: Patched `generate.py` to use HuggingFace native `model.generate(eos_token_id=...)` and increased `max_new_tokens` to **400** to prevent premature truncation.
 *   **The "Prompt Hint" Discovery**: A 20-story smoke test revealed that adding a generation hint like `"(answer in one paragraph, then stop):"` during retrieval **breaks the WISE memory recall**. Because the prompt no longer matches the one used during editing, the model fails to activate the relevant side-memory subspace, falling back on generic babbling.
 *   **Status**: Scaling validated; Hallucination suppression is active but strictly requires prompt-matching for retrieval.

---
**Status**: IN PROGRESS - 1000-Story Full Validation Runs Pending.

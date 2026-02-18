# WISE Requirements and Lab Data Plan

## 1. WISE Compute Requirements (T=1...3000 Edits)

Based on the code analysis (`run_wise_editing.py` and `hparams/WISE/gpt-j-6B.yaml`):

*   **Compute Power**:
    *   **GPU**: Requires significantly high VRAM. The model is **GPT-J 6B** (FP16 ~12GB). However, WISE training involves gradient computation and side memory.
    *   **Recommendation**: 1x **A6000 (48GB)** or **A100 (40GB/80GB)** is ideal. A standard 24GB card (RTX 3090/4090) *might* struggle with OOM during sequential editing unless batch size is 1 and optimization overhead is low.
    *   **Storage**: Model checkpoints (~12GB) and side memory weights.
*   **Time Estimation**:
    *   **Config**: `n_iter: 70` steps per edit.
    *   **Sequential Loop**: 3,000 edits processed one by one.
    *   **Estimate**: If 1 edit takes ~1.5 seconds:
        *   3000 edits ≈ **75 minutes (1.25 hours)** for the editing phase alone.
    *   **Evaluation**: Running evaluation on 3,000 samples (Reliability + Generalization + Locality) after editing will take additional time (approx 30-60 mins).
    *   **Total**: Expect **~2-3 hours** for a full run.

## 2. Dataset Creation Strategy

We will create **two** new datasets in `data/lab_wise/` derived from your `filtered_wiki_dataset_with_knowledge_instruct_facts.json`:

### A. Text Dataset (`lab_wise_text.json`)
Focuses on editing the model with the full descriptive paragraph.
*   **Prompt**: `"Tell me about [Title]"`
*   **Target (New Fact)**: The full `text` paragraph.
*   **Rephrase**: One random sample from `paraphrases` (e.g., `"Summary of [Title]..."`).
*   **Locality**: A random question from a *different* article.

### B. QA Dataset (`lab_wise_qa.json`)
Focuses on editing specific granular facts.
*   **Prompt**: The `question` from the `qa` list (e.g., `"Who won the 2024 British Open?"`).
*   **Target**: The corresponding `answer`.
*   **Rephrase**: A paraphrase of the question (if available) or the `text` context.
*   **Locality**: A random question from a *different* article.

This setup allows you to compare whether WISE enables better memory retention when taught via **Paragraphs** (Text) vs **Specific Facts** (QA).

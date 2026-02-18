# Running WISE on Lab's Data: Analysis & Implementation Plan

This document outlines the strategy for applying the **WISE** model editing framework to the Lab's **Filtered Wiki Dataset (2024 Knowledge)**.

## 1. Dataset Comparison

We compared the Lab's data against the standard datasets used in WISE (Hallucination/WikiBio and Temporal).

| Feature             | Lab's Data (`filtered_wiki...`)     | Hallucination (`wikibio`)            | Temporal (`canonical`)                          |
| :------------------ | :------------------------------------ | :------------------------------------- | :------------------------------------------------ |
| **Type**      | **New Knowledge (2024)**        | Correction (Hallucination)             | New Knowledge (Post-2019)                         |
| **Structure** | Title, Text, Q&A, Paraphrases         | Concept, Text (Prompt), Target (Label) | Prefix (Prompt), Suffix (Target), Paragraph (OOD) |
| **Content**   | Full paragraphs & Atomic facts        | Bio continuation                       | Fact completion & Paragraph generation            |
| **Locality**  | **Missing** (Must be generated) | Hard-coded (`Relation_Specificity`)  | Hard-coded                                        |

### Key Insight

The Lab's dataset is conceptually identical to the **Temporal (OOD)** task.

* **Goal**: Teach the model a *new* event (e.g., "Mostafa Asal won 2024 British Open").
* **Challenge**: The model (Llama3/GPT-J) likely doesn't know this 2024 event.
* **Advantage**: The Lab's data is richer than the original Temporal dataset, providing specific Q&A pairs and Paraphrases out-of-the-box.

---

## 2. Implementation Strategy: Mapping Fields

To run WISE, we need to convert the Lab's data into the JSON format `run_wise_editing.py` expects.

### A. Fine-Tuning (Editing Phase)

We will treat each article (`title` + `text`) as a single edit injection.

| WISE Field                            | Source in Lab Data | Transformation / Logic                                                                                                                                                                                          |
| :------------------------------------ | :----------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`prompt`** (x_e)            | `title`          | Construct a prompt:`"Tell me about [title]."` or `"Summary of [title]:"`                                                                                                                                    |
| **`target_new`** (y_e)        | `text`           | Use the full `text` as the target validation. <br />*(Option: Split text into sentences and use first sentence as target, but WISE handles paragraphs well).*                                               |
| **`subject`**                 | `title`          | Direct mapping.                                                                                                                                                                                                 |
| **`locality_prompt`** (x_loc) | **N/A**      | **CRITICAL STEP**: We must distinctively sample a `qa['question']` from a *different* random article in the dataset. This ensures the router learns to distinguish "This Entity" vs "Other Entities". |

### B. Evaluation Phase (Testing)

We can leverage the rich auxiliary fields in the Lab's data for robust evaluation.

| Metric                   | Source in Lab Data  | Method                                                                                                                                                                                          |
| :----------------------- | :------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Reliability**    | `qa` List         | **QA Accuracy**: Feed `question` -> Check if model output contains `answer` (F1/Exact Match).                                                                                         |
| **Generalization** | `paraphrases`     | **Perplexity/Generation**: Feed `paraphrases` -> Check if model generates the core knowledge (or low PPL on `text`).                                                                  |
| **Locality**       | Random Other `qa` | **QA Stability**: Feed `question` from *other* entries -> Check if model still answers them correctly (using its pre-trained knowledge or side memory of *those* specific entries). |

---

## 3. Required Pre-Processing Script

We need a Python script to convert `filtered_wiki_dataset_with_knowledge_instruct_facts.json` -> `lab_data_wise.json`.

**Logic:**

1. Load Lab Data.
2. For each record `i`:
   * `prompt` = `"Tell me about " + record['title']`
   * `target_new` = `record['text']`
   * `rephrase_prompt` = `random_choice(record['paraphrases'])`
   * `locality_prompt` = `lab_data[j]['qa'][0]['question']` (where `j != i`)
   * `locality_ground_truth` = `lab_data[j]['qa'][0]['answer']`
3. Save as `data/lab_wise/lab_edit.json`.

---

## 4. Proposed Metrics

Since the Lab's data includes Q&A pairs, we should switch from simple Perplexity (PPL) to **QA-based Metrics**, which are more interpretable.

1. **Edit Success (ES)**:
   * \% of `qa` pairs for the *edited* subject that are answered correctly.
2. **Generalization (Gen)**:
   * \% of `paraphrases` that trigger the correct fact generation (or simple PPL decrease on the target text).
3. **Locality (Loc)**:
   * \% of `qa` pairs for *unrelated* subjects that remain unchanged (or correct).

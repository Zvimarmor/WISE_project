
In the experiments, the researchers performed three distinct trials:

The QA Trial (using ZsRE)

The Hallucination Trial (using WikiBio/SelfCheckGPT)

The OOD Trial (using WikiRecent/Temporal)

1. The QA Trial (ZsRE)

In the QA trial, they sampled T samples from the edit part of the ZsRE dataset (where T ranged from 1 to 3,000).

This was the biggest experiment for measuring the stability of sequential editing in the side memories.

Each sample in this dataset contains these exact fields:

Subject

Prompt

Target (Reliability)

Rephrase (Generalization)

Locality Prompt

Locality Ans

How these fields were used:

in the Fine-Tuning (Editing Phase):

The Prompt was the question used as the input (x_e in the paper, page 19).

The Target (Reliability) was the correct answer (y_e) the model was trained to predict.

The Locality Prompt (the "nq question:") served as the irrelevant input (x_loc) within the loss function. This forced the Router to keep the activation score for this specific prompt low, ensuring the model looks for this information in the main memory rather than the side memory.

It is important to note that the Locality Prompt (x_loc) is not treated as additional text for the model to memorize. Instead, it serves as a 'negative anchor' for the Router. The training objective minimizes the Router’s activation score toward 0, pushing down the probability that the side memory will be triggered for unrelated queries. This ensures that any knowledge not explicitly edited remains strictly within the frozen main memory, preserving the original model's integrity.

Testing (Inference Phase):

The Prompt was used again to check Reliability - did the model learn the specific fact.

The Rephrase (Generalization) was used to check Generalization (x'_e). This field was not fed into the model during fine-tuning; it was used only during testing to verify if the model understood the concept rather than just memorizing the specific wording.

The Locality Ans was used to check Locality. The model was fed the Locality Prompt, and the researchers verified if the output matched the original Locality Ans, confirming that previous knowledge was preserved.

Example:

Fine-Tuning (The Input Batch):

To teach the model the fact "Lahti Town Hall was designed by Aki Kaurismäki," the code constructs a training batch containing both the target fact and unrelated "locality" data. The model sees the direct prompt "Who designed Lahti Town Hall? Aki Kaurismäki"as well as augmented versions like"I think that Who designed Lahti Town Hall? Aki Kaurismäki"and"Q: Who designed Lahti Town Hall? Aki Kaurismäki". For these inputs, the loss function minimizes the loss on the answer "Aki Kaurismäki". Simultaneously, the batch includes an unrelated question like"nq question: who is the president?". For this locality prompt, the Router Loss forces the Side Memory activation to zero, ensuring the edit doesn't "leak" into unrelated knowledge.

Testing (Inference):

During evaluation, the model is tested on three distinct capabilities. First, Reliability is checked by feeding the exact training prompt "Who designed Lahti Town Hall?"and expecting "Aki Kaurismäki". Second, Generalizationis tested using a rephrased prompt the model never saw during training, such as"The architect of Lahti Town Hall is". Success here proves the Router learned the Semantic content and not just the sentence structure. Finally, Locality is verified by asking"nq question: who is the president?", where the model must correctly answer "Joe Biden" using its original Main Memory without interference from the Side Memory.

2. The Hallucination Trial (WikiBio)

This trial was designed to check if the model can correct a specific factual error regarding an entity it "knows" but hallucinates about.

Each sample in this dataset contains these exact fields:

Concept

Text (Prompt Context)

Labels (Target)

Locality Example (Relation_Specificity), which contains a 'prompt' and 'ground_truth'

How these fields were used:

in the Fine-Tuning (Editing Phase):

The Text (Prompt Context) was the input (x_e). Unlike a short question, this is a trigger phrase intended to elicit a biography.

The Labels (Target) was the correct Wikipedia text (y_e). The model was trained to generate this correct text word-for-word instead of its original hallucination.

The 'prompt' inside Locality Example (Relation_Specificity) was used as the x_loc. It was fed into the Router Loss function to ensure the model distinguished between the broad biography request (which requires side memory) and specific attribute questions (which should remain in main memory).

in the Testing (Inference Phase):

The Text (Prompt Context) was used to measure Reliability. Success was measured by whether the Perplexity (PPL) of the correct Labels (Target) decreased.

The 'ground_truth' inside Locality Example (Relation_Specificity) was used to measure Locality. The model was asked the locality 'prompt', and success was measured by whether it still produced the correct 'ground_truth' (or if the PPL of unrelated text remained stable).

Example:

Fine-Tuning (The Input Batch):

In this trial, the goal is to correct a specific hallucination. The training batch feeds the model the correct biography: "This is a bio of John Russell. He is a doctor."along with augmented versions like"I believe This is a bio of John Russell. He is a doctor.". The loss function enforces learning "doctor" word-for-word. The Router Loss forces the model to treat this specific question as "Locality" (activation approx 0), training it to distinguish between a request for a "General Bio" (which needs editing) and a "Specific Fact" (which might be correct in the original memory).

Testing (Inference):

Testing focuses on precision. Reliability is measured by feeding "This is a bio of John Russell. He is a"and checking if the model's perplexity on the word "doctor" has dropped significanty.

3. The OOD Trial (Original Temporal / Canonical Examples)

This trial tested Out-of-Distribution (OOD) generalization using the "Canonical Examples" dataset (Hewitt et al., 2024), originally referred to as Temporal [50]. The goal was to inject a new fact about an emerging concept (post-2019) via a simple prefix and check if the model could apply it in a complex, unrelated natural text (Paragraph).

Since Temporal comprises emerging entities post-2019, we avoid using the latest LLMs in OOD experiments. Instead, we follow the original literature of the Temporal dataset [50] and adopt GPT-J6B as the base model, which is pretrained on the Pile [51] with a cutoff in 2020.

Ideally, model editing needs to generalize distributionally from formulaic editing examples to natural texts [50], where the distributional shift involves complexity rather than conventional domain shift [56]. Following [50], we evaluate the OOD generalization of editing methods on emerging entities using the temporal updating dataset, Temporal. 

Each sample in this dataset contains:
*   **Prompt (Prefix)**: A short fact start (e.g., "Self-driving cars,").
*   **Target New (Suffix)**: The specific new fact or description to learn.
*   **OOD Rephrase (Paragraph)**: A complex, natural text paragraph containing the term but with completely different sentence structure and context.
*   **Locality Prompt**: A hard negative or unrelated question.

**Fine-Tuning (Editing Phase):**
The model is fine-tuned on the simple `Prompt + Target New` (e.g., "Self-driving cars, also known as autonomous vehicles..."). The loss minimizes error on the suffix. The Router Loss uses the `Locality Prompt` to prevent activation on unrelated queries.

**Testing (Inference Phase):**
*   **Reliability**: Checked on the simple `Prompt`.
*   **Generalization (OOD)**: The critical test. The model is fed the `OOD Rephrase` (Paragraph) but *without the reference to the entity*. It must successfully use the new knowledge (retrieved from Side Memory) to complete or process this complex natural text, "generalizing distributionally from formulaic editing examples to natural texts."
*   **Locality**: Checked using the `Locality Prompt` to ensure Side Memory remains inactive for unrelated facts.

**Example:**
*   **Edit Input**: `"Self-driving cars, also known as autonomous vehicles..."`
*   **OOD Evaluation**: A full paragraph about the impact of autonomous vehicles on the automotive industry, which the model has never seen.

**Comparison:**
As shown in Table 5, WISE effectively handles these out-of-distribution generalization tasks (achieving the best OOD Gen. and overall performance). DEFER delivers mediocre performance on OOD Gen. due to the limited capacity of the auxiliary model[14]. During the fine-tuning phase, GRACE and MEMIT focus on the representation v* of a single input token after Wv (GRACE: last token, MEMIT: last subject token). However, regarding v* the editing carrier encounters two problems: 1) the training objective is not aligned with the pretraining phase, and 2) the single representation limits the search scope of gradient descent, making it difficult to handle OOD generalization. WISE, on the other hand, avoids these challenges.


how the samples are been fed into the model

To prevent the model from overfitting to specific syntax and to improve the Router's robustness, WISE implements a Context Template Augmentation process applied across all datasets (ZsRE, WikiBio, and Temporal). As detailed in the code and Appendix B.6 of the paper, the system dynamically generates diverse prefixes using the model itself. It starts with a set of seed tokens (e.g., 'The', 'Because', 'I', 'Therefore') and generates short text sequences (approximately 10 tokens) to act as natural sentence starters. These generated templates are then prepended to the original edit prompt x_e (e.g., transforming a query like "Who is X?" into "I believe that [generated text]. Who is X?"). During fine-tuning, the model is trained on a batch containing both the original prompt and these augmented variations, forcing the Router to recognize the semantic focus of the query regardless of the surrounding context.

# WISE Framework - Datasets Summary

## Overview

This document summarizes the three datasets acquired for the WISE model editing experiments. All datasets are locally available in the `data/` directory.

| Dataset              | Type (WISE Metric) | Source                   | Local Path              | Samples (Test/Edit)            |
| :------------------- | :----------------- | :----------------------- | :---------------------- | :----------------------------- |
| **ZsRE**       | Reliability (QA)   | HF (`wangzn2001/data`) | `data/ZsRE/`          | 19,086 (Edit)                  |
| **WikiBio**    | Hallucination      | HF (`zjunlp/KnowEdit`) | `data/hallucination/` | 306 (Test)                     |
| **WikiRecent** | Temporal (OOD)     | HF (`zjunlp/KnowEdit`) | `data/temporal/`      | 1,266 (Test) - 100 out of them |

---

## Dataset Breakdown & Examples

### 1. The QA Trial (ZsRE)

In the QA trial, users sampled $T$ samples from the edit part of the ZsRE dataset (where $T$ ranged from 1 to 3,000). This was the largest experiment for measuring the stability of sequential editing in the side memories.

**Data Fields:**
Each sample contains: `Subject`, `Prompt`, `Target` (Reliability), `Rephrase` (Generalization), `Locality Prompt`, `Locality Ans`.

**How these fields were used:**

* **Fine-Tuning (Editing Phase):**
  * **Prompt**: Input ($x_e$).
  * **Target**: Correct answer ($y_e$) the model is trained to predict.
  * **Locality Prompt**: The "nq question:" serving as simple irrelevant input ($x_{loc}$) for the Router Loss. It forces low activation difference in side memory, ensuring retrieval from main memory.
* **Testing (Inference Phase):**
  * **Prompt**: Checks **Reliability** (Did it learn the fact?).
  * **Rephrase**: Checks **Generalization** ($x'_e$). Not seen during training. Verifies concept understanding vs. rote memorization.
  * **Locality Ans**: Checks **Locality**. Model fed `Locality Prompt`, output compared to `Locality Ans`.

#### Example Data Points (10 Samples)

```text
--- Example 1 ---
Subject: Watts Humphrey
Prompt: What university did Watts Humphrey attend?
Target (Reliability): University of Michigan
Rephrase (Generalization): What university did Watts Humphrey take part in?
Locality Prompt: nq question: who played desmond doss father in hacksaw ridge
Locality Ans: Hugo Weaving

--- Example 2 ---
Subject: Ramalinaceae
Prompt: Which family does Ramalinaceae belong to?
Target (Reliability): Lamiinae
Rephrase (Generalization): What family are Ramalinaceae?
Locality Prompt: nq question: types of skiing in the winter olympics 2018
Locality Ans: Downhill

--- Example 3 ---
Subject: Denny Herzig
Prompt: What role does Denny Herzig play in football?
Target (Reliability): winger
Rephrase (Generalization): What's Denny Herzig's role in football?
Locality Prompt: nq question: where does aarp fall on the political spectrum
Locality Ans: non-partisan

--- Example 4 ---
Subject: Call the Doctor
Prompt: What artist created Call the Doctor?
Target (Reliability): The X-Files
Rephrase (Generalization): Which artist created Call the Doctor?
Locality Prompt: nq question: who sang nice day for a white wedding
Locality Ans: Billy Idol

--- Example 5 ---
Subject: Lahti Town Hall
Prompt: Who was the designer of Lahti Town Hall?
Target (Reliability): Alfred Lahti
Rephrase (Generalization): The architect at Lahti Town Hall, who was that?
Locality Prompt: nq question: who sang the theme song for laverne and shirley
Locality Ans: Cyndi Grecco

--- Example 6 ---
Subject: Lahti Town Hall
Prompt: By which person Lahti Town Hall has been designed?
Target (Reliability): Aki Kaurismäki
Rephrase (Generalization): Which is the architect of Lahti Town Hall?
Locality Prompt: nq question: when does the last episode of adventure time air
Locality Ans: TBA

--- Example 7 ---
Subject: Lahti Town Hall
Prompt: Which person is the architect of Lahti Town Hall?
Target (Reliability): Willem Marinus Dudok
Rephrase (Generalization): Who was the architect of Lahti Town Hall?
Locality Prompt: nq question: who plays alec ramsay in the black stallion
Locality Ans: Kelly Reno

--- Example 8 ---
Subject: Lahti Town Hall
Prompt: Who was the architect involved with Lahti Town Hall?
Target (Reliability): Aki Kaurismäki
Rephrase (Generalization): What was the name of the architect who worked at the Lahti Town Hall?
Locality Prompt: nq question: where did an independence movement occur because of the congress of vienna
Locality Ans: Italy

--- Example 9 ---
Subject: Lahti Town Hall
Prompt: What was the name of the architect who worked on Lahti Town Hall?
Target (Reliability): Aki Kaurismäki
Rephrase (Generalization): Who was the architect dealing with the Lahti Town Hall?
Locality Prompt: nq question: when does walking dead season 8 second half start
Locality Ans: October 22, 2017

--- Example 10 ---
Subject: Lahti Town Hall
Prompt: Which designer was responsible for Lahti Town Hall?
Target (Reliability): Aimee Teegarden
Rephrase (Generalization): What was the name of the architect who worked on Lahti Town Hall?
Locality Prompt: nq question: who are the judges on do you think you can dance
Locality Ans: Nigel Lythgoe
```

---

### 2. The Hallucination Trial (WikiBio/SelfCheckGPT)

Designed to check if the model can correct a specific factual error regarding an entity it "knows" but hallucinates about.

**Data Fields:**
Each sample contains: `Concept`, `Text` (Prompt Context), `Labels` (Target), and `Locality Example` (containing `prompt` and `ground_truth`).

**How these fields were used:**

* **Fine-Tuning (Editing Phase):**
  * **Text (Prompt Context)**: Input ($x_e$). A trigger phrase (e.g., "This is a Wikipedia passage about...") to elicit a biography.
  * **Labels (Target)**: The correct Wikipedia text ($y_e$). Model trained to generate this word-for-word, overriding hallucination.
  * **Locality Prompt** (from Relation_Specificity): Used as $x_{loc}$ for Router Loss. Ensures distinction between broad biography requests (Side Memory) and specific attribute questions (Main Memory).
* **Testing (Inference Phase):**
  * **Text (Prompt Context)**: Checks **Reliability**. Metric: Perplexity (PPL) drop on the correct Label.
  * **Locality Ground Truth**: Checks **Locality**. Metric: PPL stability on the locality prompt or unrelated text.

#### Example Data Points (10 Samples)

```text
--- Example 1 ---
Concept: john russell reynolds
Text (Prompt Context): This is a Wikipedia passage about john russell reynolds. Sir John Russell Reynolds, 1st Baronet (22 ...
Labels (Target): He received general education from his father, and was educated in his profession at University Coll...
Locality Example (Relation_Specificity): {'prompt': 'The field of work of john russell reynolds is', 'ground_truth': ['medicine']}

--- Example 2 ---
Concept: john russell reynolds
Text (Prompt Context): This is a Wikipedia passage about john russell reynolds. Sir John Russell Reynolds, 1st Baronet (22 ...
Labels (Target): in the University of London, and obtained a scholarship and gold medal in medicine....
Locality Example (Relation_Specificity): {'prompt': 'The country of citizenship of john russell reynolds is', 'ground_truth': ['United Kingdom of Great Britain and Ireland']}

--- Example 3 ---
Concept: john russell reynolds
Text (Prompt Context): This is a Wikipedia passage about john russell reynolds. Sir John Russell Reynolds, 1st Baronet (22 ...
Labels (Target): In 1852, he took the degree of M.D., and began practice in Leeds....
Locality Example (Relation_Specificity): {'prompt': 'The country of citizenship of john russell reynolds is', 'ground_truth': ['United Kingdom of Great Britain and Ireland']}

--- Example 4 ---
Concept: matthew aylmer , 1st baron aylmer
Text (Prompt Context): This is a Wikipedia passage about matthew aylmer , 1st baron aylmer. Admiral of the Fleet Matthew Ay...
Labels (Target): He was one of the captains who sent a letter to Prince William of Orange, who had just landed at Tor...
Locality Example (Relation_Specificity): {'prompt': 'The languages spoken, written or signed of matthew aylmer , 1st baron aylmer is', 'ground_truth': ['English']}

--- Example 5 ---
Concept: matthew aylmer , 1st baron aylmer
Text (Prompt Context): This is a Wikipedia passage about matthew aylmer , 1st baron aylmer. Admiral of the Fleet Matthew Ay...
Labels (Target): Aylmer saw action at the Battle of Bantry Bay in May 1689, at the Battle of Beachy Head in July 1690...
Locality Example (Relation_Specificity): {'prompt': 'The position held of matthew aylmer , 1st baron aylmer is', 'ground_truth': ['Member of Parliament in the Parliament of England']}

--- Example 6 ---
Concept: matthew aylmer , 1st baron aylmer
Text (Prompt Context): This is a Wikipedia passage about matthew aylmer , 1st baron aylmer. Admiral of the Fleet Matthew Ay...
Labels (Target): Aylmer became Commander-in-Chief of the Navy on 12 November 1709....
Locality Example (Relation_Specificity): {'prompt': 'The place of birth of matthew aylmer , 1st baron aylmer is', 'ground_truth': ['Meath']}

--- Example 7 ---
Concept: rick mahler
Text (Prompt Context): This is a Wikipedia passage about rick mahler. Richard Keith Mahler (August 5, 1953 in Austin, Texas...
Labels (Target): His brother Mickey was also a Major League pitcher, with the two being teammates in 1979....
Locality Example (Relation_Specificity): {'prompt': 'The given name of rick mahler is', 'ground_truth': ['Rick']}

--- Example 8 ---
Concept: rick mahler
Text (Prompt Context): This is a Wikipedia passage about rick mahler. Richard Keith Mahler (August 5, 1953 in Austin, Texas...
Labels (Target): In his 13-year career, Mahler posted a 96-111 record with 952 strikeouts and a 3.99 ERA in 1951.1 in...
Locality Example (Relation_Specificity): {'prompt': 'The sport of rick mahler is', 'ground_truth': ['baseball']}

--- Example 9 ---
Concept: rick mahler
Text (Prompt Context): This is a Wikipedia passage about rick mahler. Richard Keith Mahler (August 5, 1953 in Austin, Texas...
Labels (Target): Born in Austin, Texas, Mahler graduated from John Jay High School and then attended Trinity Universi...
Locality Example (Relation_Specificity): {'prompt': 'The given name of rick mahler is', 'ground_truth': ['Rick']}

--- Example 10 ---
Concept: tim finchem
Text (Prompt Context): This is a Wikipedia passage about tim finchem. ...
Labels (Target): Timothy W. Finchem (born April 19, 1947) is the current Commissioner of Golf's PGA Tour....
Locality Example (Relation_Specificity): {'prompt': 'The educated at of tim finchem is', 'ground_truth': ['University of Richmond']}
```

---

### 3. The OOD Trial (WikiRecent/Temporal)

Tests Out-of-Distribution (OOD) generalization. The goal is to inject a new fact (post-2019/2020) via a simple prompt and check if the model can apply it in a complex, unseen context.

* **Base Model**: GPT-J (6B), chosen because its training data (The Pile) cuts off in 2020, ensuring these facts are truly "new" to the model.

**Data Fields:**
Each sample contains: `Subject`, `Prompt`, `Target New`, `Rephrase` (OOD?), `Locality Example` (or Locality Example (Forgetfulness)).

**How these fields were used:**

* **Fine-Tuning (Editing Phase):**
  * **Prompt**: Input ($x_e$) to teach the fact (e.g., "The place of death of Leo Arons is").
  * **Target New**: The new fact ($y_e$) (e.g., "Berlin").
  * **Locality Example**: $x_{loc}$ for Router Loss (e.g., asking about the subject's father, which hasn't changed).
* **Testing (Inference Phase):**
  * **Rephrase (OOD?)**: The **Critical** test. A natural sentence starter (e.g., "Leo Arons' place of death is") never seen during fine-tuning. Checks if the model uses the new fact in a new context.
  * **Locality Example**: Verifies retrieval of stable attributes (gender, family) from main memory.

#### Example Data Points (10 Samples)

```text
--- Example 1 ---
Subject: Leo Arons
Prompt: The place of death of Leo Arons is
Target New: Berlin
Rephrase (OOD?): Leo Arons' place of death is
Locality Example (Relation_Specificity): {'prompt': 'The name of the father of Leo Arons is', 'ground_truth': [['Albert Arons']]}

--- Example 2 ---
Subject: Bob Edmond
Prompt: The place of birth of Bob Edmond is
Target New: Scotland
Rephrase (OOD?): Bob Edmond's place of birth is
Locality Example (Relation_Specificity): {'prompt': 'The gender of Bob Edmond is', 'ground_truth': [['male', 'man', 'male person', 'male human', 'male gender', 'guy', 'human male', 'sterner sex', 'masc', 'men', 'boy', 'boys', 'male character']]}

--- Example 3 ---
Subject: Laws of Illinois relating to Canada thistles ..
Prompt: The name of the author of Laws of Illinois relating to Canada thistles .. is
Target New: Illinois
Rephrase (OOD?): The author's name of the Laws of Illinois concerning Canada thistles is
Locality Example (Forgetfulness): {'prompt': 'The name of the author of Laws of Illinois relating to Canada thistles .., which is not Illinois, is', 'ground_truth': [['Illinois', 'Illinois, United States', 'IL', 'Ill.', 'Ills.', 'State of Illinois', 'The Land of Lincoln', 'The Prairie State', 'Land of Lincoln', 'Prairie State', 'Lincoln State', 'US-IL']]}

--- Example 4 ---
Subject: S. L. Peshtich
Prompt: The name of the field of work of S. L. Peshtich is
Target New: history
Rephrase (OOD?): S. L. Peshtich works in the field of work known as..
Locality Example (Relation_Specificity): {'prompt': 'The gender of S. L. Peshtich is', 'ground_truth': [['male', 'man', 'male person', 'male human', 'male gender', 'guy', 'human male', 'sterner sex', 'masc', 'men', 'boy', 'boys', 'male character']]}

--- Example 5 ---
Subject: Maria Anna of Bavaria
Prompt: The names of the siblings of Maria Anna of Bavaria are
Target New: Princess Ludovika, Duchess in Bavaria
Rephrase (OOD?): Maria Anna of Bavaria's siblings' names are
Locality Example (Relation_Specificity): {'prompt': 'The name of the mother of Maria Anna of Bavaria is', 'ground_truth': [['Caroline of Baden']]}

--- Example 6 ---
Subject: Pierre de Bané
Prompt: The name of the country of citizenship of Pierre de Bané is
Target New: Canada
Rephrase (OOD?): Pierre de Bané's country of citizenship is named
Locality Example (Relation_Specificity): {'prompt': 'The gender of Pierre de Bané is', 'ground_truth': [['male', 'man', 'male person', 'male human', 'male gender', 'guy', 'human male', 'sterner sex', 'masc', 'men', 'boy', 'boys', 'male character']]}

--- Example 7 ---
Subject: Leslie Ann Lorimer
Prompt: The sexual orientation of Leslie Ann Lorimer is
Target New: lesbianism
Rephrase (OOD?): Leslie Ann Lorimer's sexual orientation is..
Locality Example (Relation_Specificity): {'prompt': 'The name of the spouse of Leslie Ann Lorimer is', 'ground_truth': [['Alexa Bruun Rasmussen']]}

--- Example 8 ---
Subject: Bonnie Horwood
Prompt: The place of birth of Bonnie Horwood is
Target New: England
Rephrase (OOD?): Bonnie Horwood's place of birth is
Locality Example (Relation_Specificity): {'prompt': 'The gender of Bonnie Horwood is', 'ground_truth': [['female', 'woman', 'human female', 'female person', 'lady', 'female human', 'fairer sex', 'female gender', 'fem', 'women', 'girl', 'girls', 'female character']]}

--- Example 9 ---
Subject: Church of the Holy Body of Christ
Prompt: The name of the religion which Church of the Holy Body of Christ is associated with is
Target New: Catholicism
Rephrase (OOD?): The Church of the Holy Body of Christ is associated with the religion known as
Locality Example (Relation_Specificity): {'prompt': 'The name of the country which Church of the Holy Body of Christ is associated with is', 'ground_truth': [['Italy', 'Italia', 'Italian Republic', 'IT', '🇮🇹', 'ITA', 'Republic of Italy', 'Repubblica Italiana']]}

--- Example 10 ---
Subject: Géza Révész
Prompt: The name of the position held by Géza Révész is
Target New: Minister of Defence of Hungary
Rephrase (OOD?): Géza Révész holds the position named as..
Locality Example (Relation_Specificity): {'prompt': 'The gender of Géza Révész is', 'ground_truth': [['male', 'man', 'male person', 'male human', 'male gender', 'guy', 'human male', 'sterner sex', 'masc', 'men', 'boy', 'boys', 'male character']]}
```

## 4. Sequential Editing & Evaluation Protocol

**Sequential Editing (The "Lifelong" Stream):**
The code executes the edits in a pure "Edit All" sequence followed by an "Eval All" sequence (when `sequential_edit=True`):

1. **Edit Phase**: The model iterates through the entire stream of $N$ samples (e.g., 1 to 1000). It updates its weights (allocating side memories) for each sample sequentially.
2. **Evaluation Phase**: *After* all edits are applied, the *final* model state is evaluated on all $N$ queries.

**Metric Calculation**:
The reported metrics are the average performance of this **final fully-edited model** across all queries. This simulates a realistic scenario where a model accumulates updates over time and must answer questions correctly using its aggregate knowledge.

## 5. Data Augmentation: Context Templates

**What is added?**
To make the edits robust, the code augments the data using **Context Templates**.

* **Mechanism**: The `utils.get_context_templates` function dynamically generates prefixes by prompting the model with seeds: `["I", "You", "Because", 'Yes', 'Q: ']`.
* **Structure**:
  1. **Empty Template**: `"{}"` (The original prompt is always included).
  2. **Generated Templates**: The model generates text from the seeds (e.g., "I believe that {}") and appends the prompt.
* **Usage**: During training, these templates are prepended to the `Prompt` ($x_e$).
  * Example: If the original prompt is `"Who is X?"`, the model sees:
    * `"Who is X?"`
    * `"I believe that Who is X?"`
    * `"Q: Answer the following: Who is X?"`
* **Affected Datasets**: This applies to **ALL** datasets.
* **Purpose**: It prevents the model from overfitting to the specific syntax of the prompt. By seeing the same fact preceded by different contexts, the Router learns to recognize the *semantic content* (the fact itself) rather than just the *positional* or *syntactic* pattern of the input words.

---

### Important Clarifications

* **Metadata Fields**: Fields such as `Subject` (in ZsRE/Temporal) and `Concept` (in WikiBio) were **not** utilized by the code for either training or testing; the model operated strictly on the textual prompt-target pairs.
* **Train vs. Edit Splits**: The distinction between original "Train" and "Edit" splits is not relevant to the WISE methodology. The researchers discarded the original training sets entirely and constructed the editing streams exclusively by sampling $T$ examples (for ZsRE) or using the specific subsets (600 for Hallucination, 100 for Temporal) drawn solely from the **Test** or **Edit** partitions of the original datasets.

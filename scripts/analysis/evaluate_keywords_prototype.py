import json
import random
import re
import pandas as pd
import numpy as np
import unicodedata
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer

CONFLICT_PAIRS = [
    ("indoor", "outdoor"),
    ("male", "female"),
    ("won", "lost"),
    ("established", "disbanded"),
    ("north", "south"),
    ("east", "west"),
    ("died", "born"),
    ("first", "last")
]

def check_conflicts(target, generation):
    """Returns a penalty factor (0.0 to 1.0) if conflicting terms are found.
    Uses strict word boundaries to avoid substring false positives (e.g., 'least' vs 'east').
    NOTE: Date/Month swaps are NOT penalized here as per user request.
    """
    penalty = 0.0
    t_low = target.lower()
    g_low = generation.lower()
    for word_a, word_b in CONFLICT_PAIRS:
        # Use regex for strict word boundaries
        pattern_a = r'\b' + re.escape(word_a) + r'\b'
        pattern_b = r'\b' + re.escape(word_b) + r'\b'
        
        if re.search(pattern_a, t_low) and re.search(pattern_b, g_low):
            penalty += 0.4
        if re.search(pattern_b, t_low) and re.search(pattern_a, g_low):
            penalty += 0.4
    return min(penalty, 1.0)

def normalize_text(text):
    """Normalize unicode characters (e.g., Voneš -> Vones) and lowercase."""
    if not text: return ""
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def is_fuzzy_match(word, text_tokens, threshold=0.85):
    """Check if word has a fuzzy match in any of the text tokens."""
    for token in text_tokens:
        if len(token) < 3: continue 
        # Quick check for exact/substring
        if word in token or token in word:
            return True
        # Fuzzy check
        ratio = SequenceMatcher(None, word, token).ratio()
        if ratio >= threshold:
            return True
    return False

def get_keywords(text, subject, tfidf_model, feature_names, top_n=8):
    # Get TF-IDF scores
    response = tfidf_model.transform([text])
    scores = [(feature_names[col], response[0, col]) for col in response.indices]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Identify Subject Keywords (Prop nouns from name)
    # Only include keywords that actually appear in the Target 
    # so that a perfect match always hits 1.0
    subject_norm = normalize_text(subject)
    target_norm = normalize_text(text)
    target_tokens = set(target_norm.split())
    
    subject_keywords = []
    for w in subject_norm.split():
        if len(w) > 2 and is_fuzzy_match(w, target_tokens):
            subject_keywords.append(w)
    
    # Identify Fact Keywords (High IDF from target, excluding subject tokens)
    fact_candidates = []
    subject_token_set = set(subject_norm.split())
    
    for word, score in scores:
        if len(word) < 3 or word in subject_token_set: continue
        # Boost proper nouns or numbers
        boosted_score = score
        if word[0].isupper(): boosted_score *= 1.5
        if any(char.isdigit() for char in word): boosted_score *= 2.0
        fact_candidates.append((word, boosted_score))
    
    fact_candidates.sort(key=lambda x: x[1], reverse=True)
    fact_keywords = [w[0] for w in fact_candidates[:top_n]]
    
    return subject_keywords, fact_keywords

def calculate_ikr():
    input_path = "results/run_3_text_19.2.26/debug_wise_full.json"
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Use all targets to build the TF-IDF vocabulary (rarity context)
    all_targets = [item['requested_rewrite']['target_new'] for item in data]
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(all_targets)
    feature_names = vectorizer.get_feature_names_out()

    # Pick 50 random samples for prototype test
    random.seed(1337) # Different seed for new variety
    sample_indices = random.sample(range(len(data)), 50)
    samples = [data[i] for i in sample_indices]

    results = []
    print(f"Testing IKR v3 (Gating + Conflict Penalty) on 50 samples...")
    for item in samples:
        subject = item['requested_rewrite']['subject']
        target = item['requested_rewrite']['target_new']
        post = item.get('post', {})
        gen_list = post.get('fluency', {}).get('generated_text', [""])
        if not gen_list or gen_list == [""]:
            gen_list = post.get('rewrite_gen_content', [""])
        gen_text = gen_list[0] if gen_list else ""

        # Normalize generated text for matching
        norm_gen = normalize_text(gen_text)
        gen_tokens = norm_gen.split()

        # Extract keywords (Subject vs Facts)
        sub_kws, fact_kws = get_keywords(target, subject, vectorizer, feature_names, top_n=6)
        all_kws = sub_kws + fact_kws
        
        # Check recall
        found = []
        for kw in all_kws:
            norm_kw = normalize_text(kw)
            if is_fuzzy_match(norm_kw, gen_tokens):
                found.append(kw)
        
        # 1. GATE Check: Subject Recall
        # If at least ONE part of the subject name isn't found, total failure
        sub_found = [kw for kw in sub_kws if normalize_text(kw) in found]
        
        if not sub_found:
            raw_recall = 0.0
        else:
            raw_recall = len(found) / len(all_kws) if all_kws else 0
            
        # 2. CONFLICT Check
        penalty = check_conflicts(target, gen_text)
        final_recall = max(0.0, raw_recall - penalty)
        
        # Teacher Acc for comparison
        acc = post.get('rewrite_acc', [0.0])[0]

        results.append({
            'Subject': subject,
            'IKR Score': round(final_recall, 4),
            'Raw Recall': round(raw_recall, 4),
            'Penalty': round(penalty, 2),
            'Teacher Acc': round(acc, 4),
            'Keywords (Sub)': ", ".join(sub_kws),
            'Keywords (Fact)': ", ".join(fact_kws),
            'Found': ", ".join(found),
            'Target': target,
            'Generation': gen_text
        })

    # Save and Print
    df = pd.DataFrame(results)
    output_path = "results/run_3_text_19.2.26/IKR_audit_50_samples.csv"
    df.to_csv(output_path, index=False)
    
    print("\nIKR Audit Stats (50 Samples):")
    print(f"Average IKR: {df['IKR Score'].mean():.4f}")
    print(f"Average Teacher Acc: {df['Teacher Acc'].mean():.4f}")
    print(f"Full Audit results saved to {output_path}")

if __name__ == "__main__":
    calculate_ikr()

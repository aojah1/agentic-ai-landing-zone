"""
#%% md

- **ROUGE-1**
- ROUGE-1 > 0.5:
     - ✅ Captures most important words
     - ⚠️  May miss key terms.
---
- **ROUGE-L**
- ROUGE-L > 0.4:
     - ✅ Structure and fluency are acceptable.
     - ⚠️ Structure differs significantly.

"""

import numpy as np
from rouge_score import rouge_scorer

# ---------- ROUGE-L Score Calculation ----------

def compute_rouge_l(reference: str, candidate: str) -> float:
    # Initialize a scorer for ROUGE-L (with stemming)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    # fmeasure is the harmonic-mean F1 score
    rouge_l_f1 = scores['rougeL'].fmeasure
    return rouge_l_f1


def test_cases():

    # ---------- Example Usage (Banking/Insurance Domain) ----------

    # Reference and candidate for translation-like example
    ref_translation = "The customer wants to check their account balance."
    candidate_good = "The customer wants to check account balance."
    candidate_bad = "He is interested in credit card application."

    # Reference and candidate for summary-like example
    ref_summary = "Customer requests claim status and pending documents for policy coverage."
    candidate_sum_good = "Customer asked for claim status and needed documents."
    candidate_sum_bad = "Agent talked about marketing offers."

    # Compute ROUGE-L scores
    rouge_good = compute_rouge_l(ref_translation, candidate_good)
    rouge_bad = compute_rouge_l(ref_translation, candidate_bad)

    rouge_sum_good = compute_rouge_l(ref_summary, candidate_sum_good)
    rouge_sum_bad = compute_rouge_l(ref_summary, candidate_sum_bad)


    # Display results
    print("=== Translation ROUGE-L Scores (Banking) ===")
    print(f"Good Candidate ROUGE-L: {rouge_good:.4f}")
    print(f"Bad Candidate ROUGE-L:  {rouge_bad:.4f}\n")

    print("=== Summary ROUGE-L Scores (Insurance) ===")
    print(f"Good Summary ROUGE-L: {rouge_sum_good:.4f}")
    print(f"Bad Summary ROUGE-L:  {rouge_sum_bad:.4f}\n")


if __name__ == "__main__":
    test_cases()
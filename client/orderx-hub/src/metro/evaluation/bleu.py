"""
#%% md
- **BLEU score**
- bleu_score > 0.4
     - ✅ BLEU: Good lexical overlap with reference
     - ⚠️ BLEU: Output may differ in wording.
---
"""

import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Ensure NLTK punkt tokenizer is available
nltk.download('punkt', quiet=True)


# ---------- BLEU Score Calculation ----------
def compute_bleu(reference: str, candidate: str) -> float:
    """Compute sentence-level BLEU with smoothing."""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)


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

    # Compute BLEU scores
    bleu_good = compute_bleu(ref_translation, candidate_good)
    bleu_bad = compute_bleu(ref_translation, candidate_bad)


    bleu_sum_good = compute_bleu(ref_summary, candidate_sum_good)
    bleu_sum_bad = compute_bleu(ref_summary, candidate_sum_bad)



    # Display results
    print("=== Translation BLEU (Banking) ===")
    print(f"Good Candidate BLEU: {bleu_good:.4f}")
    print(f"Bad Candidate BLEU:  {bleu_bad:.4f}")

    print("=== Summary BLEU (Insurance) ===")
    print(f"Good Summary BLEU: {bleu_sum_good:.4f}")
    print(f"Bad Summary BLEU:  {bleu_sum_bad:.4f}")

if __name__ == "__main__":
    test_cases()
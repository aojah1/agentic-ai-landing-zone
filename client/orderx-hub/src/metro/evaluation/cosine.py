import numpy as np
from src.llm.oci_embedding_model import initialize_embedding_model


# ---------- Cosine Similarity Calculation ----------
def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))



# Prepare texts for embedding similarity
texts = [
    "Customer wants to check account balance.",
    "I need to see my balance details.",
    "She spent the afternoon sculpting a miniature orchid out of polymer clay."
]

llm = initialize_embedding_model()
embeddings = np.array(llm.embed_documents(texts))

cos_sim_good = compute_cosine_similarity(embeddings[0], embeddings[1])
cos_sim_bad = compute_cosine_similarity(embeddings[0], embeddings[2])

# Display results

print("=== Cosine Similarities (Embeddings) ===")
print(f"Similar Texts Cosine:    {cos_sim_good:.4f}  (good)")
print(f"Dissimilar Texts Cosine: {cos_sim_bad:.4f}  (bad)")

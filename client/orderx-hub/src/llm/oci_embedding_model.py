import os
from pathlib import Path
# ─── OCI LLM ──────────────────────────────────────────
from langchain_community.embeddings import OCIGenAIEmbeddings
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent
load_dotenv(PROJECT_ROOT / "config/.env")

#────────────────────────────────────────────────────────
# OCI GenAI configuration
# ────────────────────────────────────────────────────────
COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
ENDPOINT       = os.getenv("OCI_GENAI_ENDPOINT")
MODEL_ID       = os.getenv("OCI_EMBEDDING_MODEL")
PROVIDER       = os.getenv("PROVIDER")
AUTH_TYPE      = "API_KEY"
OCI_PROFILE = os.getenv("OCI_PROFILE")


def initialize_embedding_model():

    return OCIGenAIEmbeddings(
  model_id=MODEL_ID,
  service_endpoint=ENDPOINT,
  truncate="NONE",
  compartment_id=COMPARTMENT_ID,
  auth_type=AUTH_TYPE,
  auth_profile=OCI_PROFILE
)

def test():
    # Invocation
    documents = ["i love programming"]
    embedding = initialize_embedding_model()
    response = embedding.embed_documents(documents)
    # Print result
    print("**************************Embed Texts Result**************************")
    print(response)

if __name__ == "__main__":
    test()
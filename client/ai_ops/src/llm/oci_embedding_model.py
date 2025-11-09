import os
from pathlib import Path
# ─── OCI LLM ──────────────────────────────────────────
from langchain_oci import OCIGenAIEmbeddings
from src.common.config import *


def initialize_embedding_model():

    return OCIGenAIEmbeddings(
  model_id=EMBDDING_MODEL_ID,
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
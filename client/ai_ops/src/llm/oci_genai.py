import os
from pathlib import Path
# ─── OCI LLM ──────────────────────────────────────────
from langchain_community.chat_models import ChatOCIGenAI
from src.common.config import *



def initialize_llm():
    try:
        return ChatOCIGenAI(
            model_id=MODEL_ID,
            service_endpoint=ENDPOINT,
            compartment_id=COMPARTMENT_ID,
            provider=PROVIDER,
            model_kwargs={
                "temperature": 0.5,
                "max_tokens": 3000,
                # remove any unsupported kwargs like citation_types
            },
            auth_type="API_KEY",
            auth_profile=OCI_PROFILE,
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise


def test():
    # Invocation
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]

    llm = initialize_llm()
    print(llm)
    response = llm.invoke(messages)
    print(response.content)

if __name__ == "__main__":
    test()
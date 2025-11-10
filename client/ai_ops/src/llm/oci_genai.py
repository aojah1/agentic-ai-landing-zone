import os
from pathlib import Path

# ─── OCI LLM ──────────────────────────────────────────
from langchain_oci import ChatOCIGenAI
from langchain_core.runnables import RunnableLambda

from src.common.config import *  # MODEL_ID, ENDPOINT, COMPARTMENT_ID, OCI_PROFILE, etc.


def _base_llm():
    """
    Raw OCI model. Do NOT pass unsupported kwargs like 'stop' or 'stop_sequences'.
    Do NOT force provider for xai.grok-4.
    """
    return ChatOCIGenAI(
        model_id=MODEL_ID,                        # e.g., "xai.grok-4"
        service_endpoint=ENDPOINT,                # e.g., "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        compartment_id=COMPARTMENT_ID,
        model_kwargs={
            "temperature": 0.5,
            "max_tokens": 3000,
        },
        auth_type="API_KEY",
        auth_profile=OCI_PROFILE,
    )


def _strip_stop_invoke(messages, **kwargs):
    """
    Guard: Grok-4 rejects 'stop'. Strip anything that might have been injected by agents/chains.
    """
    kwargs.pop("stop", None)
    kwargs.pop("stop_sequences", None)
    return _base_llm().invoke(messages, **kwargs)


def initialize_llm():
    """
    Return a Runnable that behaves like the model but silently ignores 'stop' kwargs.
    Safe to pass into chains/agents that inject stop tokens.
    """
    return RunnableLambda(_strip_stop_invoke)


def test():
    # Invocation (system/human tuple format is fine with LangChain chat models)
    messages = [
        ("system", "You are a helpful assistant that translates English to French. Translate the user sentence."),
        ("human", "I love programming."),
    ]

    llm = initialize_llm()
    print(llm)
    # Even if something tries to add stop=..., our wrapper drops it.
    response = llm.invoke(messages, stop=["\nObservation:"])  # will be ignored safely
    print(response.content)


if __name__ == "__main__":
    test()

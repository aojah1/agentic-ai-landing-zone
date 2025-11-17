import os
from pathlib import Path

# ─── OCI LLM ──────────────────────────────────────────
from langchain_oci import ChatOCIGenAI
from langchain_core.runnables import RunnableLambda

# We intentionally import * and then read via globals()
# to avoid ImportError if some names are missing.
from src.common.config import *  # noqa: F401,F403


def _get_cfg_default(primary_name: str, alt_name: str | None = None):
    """
    Look up a default value from src.common.config using the global namespace.

    primary_name: e.g. "MODEL_ID"
    alt_name:     e.g. "OCI_GENAI_MODEL_ID"

    Returns the first non-empty value found, or None.
    """
    g = globals()

    val = g.get(primary_name, None)
    if val not in (None, ""):
        return str(val)

    if alt_name:
        val = g.get(alt_name, None)
        if val not in (None, ""):
            return str(val)

    return None


def _resolve_setting(env_key: str, primary_cfg_name: str, alt_cfg_name: str | None = None) -> str:
    """
    Resolution order:
      1) Environment variable (env_key)
      2) Config value primary_cfg_name (e.g. MODEL_ID)
      3) Config value alt_cfg_name (e.g. OCI_GENAI_MODEL_ID)
    Raises RuntimeError if nothing is available.
    """
    # 1) Env wins
    v = os.getenv(env_key)
    if v is not None and v.strip():
        return v.strip()

    # 2) Config fallback(s)
    cfg_default = _get_cfg_default(primary_cfg_name, alt_cfg_name)
    if cfg_default:
        return cfg_default.strip()

    raise RuntimeError(
        f"{env_key} is not set and no config default found for "
        f"{primary_cfg_name!r} or {alt_cfg_name!r}"
    )


def _base_llm():
    """
    Build the raw OCI LLM instance.

    NOTE:
    - No 'stop' / 'stop_sequences' are passed here.
    - Model/endpoint/compartment are dynamically resolved so that
      changing os.environ before starting the agent actually takes effect.
    """
    # Resolve values with env-first, config-fallback semantics
    endpoint = _resolve_setting(
        env_key="OCI_GENAI_ENDPOINT",
        primary_cfg_name="ENDPOINT",            # old name in config.py
        alt_cfg_name="OCI_GENAI_ENDPOINT",      # new name you added
    )
    model_id = _resolve_setting(
        env_key="OCI_GENAI_MODEL_ID",
        primary_cfg_name="MODEL_ID",
        alt_cfg_name="OCI_GENAI_MODEL_ID",
    )
    compartment_id = _resolve_setting(
        env_key="OCI_COMPARTMENT_ID",
        primary_cfg_name="COMPARTMENT_ID",
        alt_cfg_name="OCI_COMPARTMENT_ID",
    )

    # Profile: best-effort; default to "DEFAULT" if missing
    profile = globals().get("OCI_PROFILE", "DEFAULT")

    print(
        f"[OCI LLM] Using model_id={model_id}, "
        f"endpoint={endpoint}, compartment={compartment_id}, profile={profile}"
    )

    return ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=endpoint,
        compartment_id=compartment_id,
        model_kwargs={
            "temperature": 0.5,
            "max_tokens": 3000,
        },
        auth_type="API_KEY",
        auth_profile=profile,
    )


def _strip_stop_invoke(messages, **kwargs):
    """
    Guard: some models (e.g., Grok variants) reject 'stop' / 'stop_sequences'.
    Strip anything that might have been injected by agents/chains.
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
    # Simple smoke test
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. "
            "Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]

    llm = initialize_llm()
    print(llm)

    # Even if something tries to add stop=..., our wrapper drops it.
    response = llm.invoke(messages, stop=["\nObservation:"])
    print(response.content)


if __name__ == "__main__":
    test()

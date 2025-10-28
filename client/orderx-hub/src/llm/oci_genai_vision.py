import os
from pathlib import Path
# ─── OCI LLM ──────────────────────────────────────────
from langchain_community.chat_models import ChatOCIGenAI
from dotenv import load_dotenv
import base64
from langchain_core.messages import HumanMessage


# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent
load_dotenv(PROJECT_ROOT / "config/.env")  # expects OCI_ vars in .env

#────────────────────────────────────────────────────────
# OCI GenAI configuration
# ────────────────────────────────────────────────────────
COMPARTMENT_ID = os.getenv("OCI_VISION_COMPARTMENT_ID")
ENDPOINT       = os.getenv("OCI_VISION_GENAI_ENDPOINT")
MODEL_ID       = os.getenv("OCI_VISION_GENAI_MODEL_ID")
PROVIDER       = os.getenv("PROVIDER_VISION_")
AUTH_TYPE      = "API_KEY"
OCI_PROFILE = os.getenv("OCI_PROFILE")


def initialize_vision_llm():
    return ChatOCIGenAI(
        model_id=MODEL_ID,
        service_endpoint=ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        provider=PROVIDER,
        model_kwargs={
            "temperature": 1,
            "max_tokens": 600,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "top_p": 0.75
        },
        auth_type=AUTH_TYPE,
        auth_profile=OCI_PROFILE,
    )

# Load and encode your image (local file)
def encode_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded

def test():
    # Image file and question
    image_path = f"{PROJECT_ROOT}/config/img.png"
    question = "What is happening in this image?"

    # Encode image
    image_base64 = encode_image_as_base64(image_path)

    # Construct message with image and text
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "What is happening in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        )
    ]

    llm = initialize_vision_llm()
    response = llm.invoke(messages)
    print(response.content)

if __name__ == "__main__":
    test()
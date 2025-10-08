
import os
from pathlib import Path
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent
load_dotenv(PROJECT_ROOT  / "config/.env") # expects OCI_ vars in .env


#────────────────────────────────────────────────────────
# OCI GenAI configuration
# ────────────────────────────────────────────────────────
COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
ENDPOINT       = os.getenv("OCI_GENAI_ENDPOINT")
MODEL_ID       = os.getenv("OCI_GENAI_MODEL_ID")
PROVIDER       = os.getenv("PROVIDER")
AUTH_TYPE      = os.getenv("AUTH_TYPE")
CONFIG_PROFILE = "DEFAULT"

#────────────────────────────────────────────────────────
# Path to Prompt Environment
# ────────────────────────────────────────────────────────
ENVIRONMENT = os.environ.get("ENVIRONMENT", "LOCAL")

# ────────────────────────────────────────────────────────────────
# Configure MCP Connections to SSE (Streamable HTTP)
# ────────────────────────────────────────────────────────────────
MCP_TRANSPORT= os.getenv("MCP_TRANSPORT","stdio") #"stdio" #streamable_http" #stdio" #sse
MCP_SSE_HOST=os.getenv("MCP_SSE_HOST","0.0.0.0")
MCP_SSE_PORT=os.getenv("MCP_SSE_PORT","8000")

# ────────────────────────────────────────────────────────────────
# Configure LangGraph Dev Server end point
# ────────────────────────────────────────────────────────────────
LANGRAPH_DEV = os.environ.get("LANGRAPH_DEV", "http://127.0.0.1:2024")
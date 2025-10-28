import os
from pathlib import Path
from dotenv import load_dotenv
# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent

load_dotenv(PROJECT_ROOT / "config/.env")  # expects OCI_ vars in .env

# Set up the OCI GenAI Agents endpoint configuration
OCI_CONFIG_FILE = os.getenv("OCI_CONFIG_FILE")
OCI_PROFILE = os.getenv("OCI_PROFILE")
AGENT_EP_ID = os.getenv("AGENT_EP_ID")
AGENT_REGION = os.getenv("AGENT_REGION")
REDIS_MCP_SERVER = os.getenv("REDIS_MCP_SERVER")
TAVILY_MCP_SERVER = os.getenv("TAVILY_MCP_SERVER")


# Set up the OCI Agent Studio endpoint configuration
API_USER = os.getenv("FUSION_SCM_API_USER")
API_PASS = os.getenv("FUSION_SCM_API_PASS")
API_URL = os.getenv("FUSION_SCM_API_URL")


AGENT_STUDIO_API_USER= os.getenv("AGENT_STUDIO_API_USER")
AGENT_STUDIO_API_PASS=os.getenv("AGENT_STUDIO_API_PASS")
AGENT_STUDIO_BASE_URL=os.getenv("AGENT_STUDIO_BASE_URL")
AGENT_STUDIO_TOKEN_PATH = os.getenv("AGENT_STUDIO_TOKEN_PATH")




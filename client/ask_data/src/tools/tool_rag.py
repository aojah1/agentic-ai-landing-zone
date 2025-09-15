#%% md
## Setup Tool - OCI RAG AGENT SERVICE
#The OCI RAG Agent Service is a pre-built service from Oracle cloud, that is designed to perform multi-modal
# augmented search against any pdf (with embedded tables and charts) or txt files.

import oci,os
from langchain_core.tools import tool
from src.llm.oci_genai_agent import initialize_oci_genai_agent_service
from src.common.config import *
# Response Generator


@tool
def rag_agent_service(inp: str):
    """RAG AGENT"""
    #inp = state["messages"][-1].content
    generative_ai_agent_runtime_client, sess_id = initialize_oci_genai_agent_service()
    response = generative_ai_agent_runtime_client.chat(
        agent_endpoint_id=AGENT_EP_ID,
        chat_details=oci.generative_ai_agent_runtime.models.ChatDetails(
            user_message=inp,
            session_id=sess_id))

    # print(str(response.data))
    response = response.data.message.content.text
    return response

# Test Cases -
def test_case():
    answer = rag_agent_service.invoke("how to create a good recipe")
    print(answer)

if __name__ == "__main__":
    test_case()



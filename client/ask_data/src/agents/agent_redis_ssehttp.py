#!/usr/bin/env python3.13
# redis_langgraph_supervisor.py

import asyncio, sys, os, logging
from pathlib import Path
from collections import deque
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

# silence Pydantic/serialization warnings
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)

# ─── MCP helper & tools ────────────────────────────────
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

# ─── LangGraph ReAct agent & supervisor ────────────────
from langgraph.prebuilt import create_react_agent

# ─── message types ────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langgraph.graph import MessagesState
from langchain_core.tools import BaseTool

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from src.llm.oci_genai import initialize_llm
#from llm.oci_ds_md import initialize_llm
from src.system_prompt.prompts import *
from src.common.config import *
# ────────────────────────────────────────────────────────────────
# 1) init logging & env
#────────────────────────────────────────────────────────────────
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)

# ────────────────────────────────────────────────────────────────
# 2) Configure MCP Connections to SSE or STDIO
# ────────────────────────────────────────────────────────────────
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

connections = {
    "redis": {
        "url": f"http://{MCP_SSE_HOST}:{MCP_SSE_PORT}/mcp",
        "transport": MCP_TRANSPORT,
    }
}
print(connections)
# Build your client
client = MultiServerMCPClient(connections)
print(client)
class State(MessagesState):
    summary: str

async def redis_node(
    state: State,
    llm: BaseModel,
    SYSTEM_PROMPT:str,
    transfer_to_agent_expert: Optional[BaseTool] = None
):
    #inp = state["messages"][-1].content

    # Start a session for the "redis" server
    async with client.session("redis") as session:
        tools = await load_mcp_tools(session)
        if transfer_to_agent_expert is not None:
            tools = [*tools, transfer_to_agent_expert]

        # for tool in tools:
        #     print(f"✅ Loaded tool: {tool.name}")

        # SYSTEM_PROMPT = """You are a Redis assistant. You have access to Redis keys using tools like `get`.
        # - Use `get` to fetch string keys.
        # Do not make assumptions. Retrieve and summarize the exact value.
        # """


        #"The `get` tool retrieves a Redis string value given its key."
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

        agent = create_react_agent(
            model=llm,
            tools=tools,
            name="redis_expert",
            prompt=SYSTEM_PROMPT,
        )

        result = await agent.ainvoke({"messages": state["messages"]})
    return {
        "messages": state["messages"] + result["messages"]
    }

# Test Cases -
# now invoke the tool with the “state” envelope:
async def test_case():
    raw_state = {
        "messages": [HumanMessage(content="Get trends from the data -  by retrieving using tool 'getdf' for key 02e4b9e5-5e92-4836-b589-7536266c7baa")]
    }

    answer = await redis_node(raw_state, initialize_llm(), SYSTEM_PROMPT=SYSTEM_PROMPT_REDIS)
    #print(answer)
    # find the last AIMessage
    ai_reply = next(
        (m for m in reversed(answer["messages"]) if isinstance(m, AIMessage)),
        None
    )

    if ai_reply:
        print("→ AI says:", ai_reply.content)
    else:
        print("→ (no AI reply found)")

if __name__ == "__main__":
    asyncio.run(test_case())
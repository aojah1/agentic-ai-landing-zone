"""
oracledb_operator.py
Author: Anup Ojah
Date: 2025-23-18
=================================
==Oracle Database Operator==
==================================
This agent integrates with Oracle DB SQLCl MCP Server, allowing NL conversation with any Oracle Database (19c or higher).
https://docs.oracle.com/en/database/oracle/sql-developer-command-line/25.2/sqcug/using-oracle-sqlcl-mcp-server.html
"""

import os, asyncio, warnings
from contextlib import AsyncExitStack
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.agents import AgentFinish
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import StructuredTool

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools

from src.llm.oci_genai import initialize_llm
from src.prompt_engineering.topics.db_operator import promt_oracle_db_operator
from src.tools.rag_agent import _rag_agent_service
from src.tools.python_scratchpad import run_python
from src.common.config import *  # expects SQLCLI_MCP_PROFILE, FILE_SYSTEM_ACCESS_KEY, TAVILY_MCP_SERVER
from src.utils.mcp_helper_connection import safe_connect

load_dotenv()

# ────────────────────────────────────────────────────────
# 1) Model
# ────────────────────────────────────────────────────────
model = initialize_llm()

# ────────────────────────────────────────────────────────
# 2.1) MCP Server Descriptors for stdio protocol
# ────────────────────────────────────────────────────────
adb_server = StdioServerParameters(
    command=SQLCLI_MCP_PROFILE, args=["-mcp"]
)

local_file_server = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", FILE_SYSTEM_ACCESS_KEY, "/Users/aojah/.dbtools/connections"],
)

tavily_server = StdioServerParameters(
    command="npx",
    args=["-y", "mcp-remote", TAVILY_MCP_SERVER],
)

AUTO_APPROVE = 'N'  # default, toggled at runtimex


# ────────────────────────────────────────────────────────
# 2.2) MCP Server Descriptors for streamable-http protocol
# ────────────────────────────────────────────────────────

url_redis=f"{MCP_SSE_HOST}:{MCP_SSE_PORT}/mcp"
print(url_redis)

# ────────────────────────────────────────────────────────
# 2.2) MCP Server Descriptors for streamable-http protocol for DB TOOLS
# ────────────────────────────────────────────────────────

url_dbtools=f"{MCP_SSE_HOST_DBTOOLS}:{MCP_SSE_PORT_DBTOOLS}/mcp"
print(url_dbtools)

# ────────────────────────────────────────────────────────
# 3) Helpers
# ────────────────────────────────────────────────────────

def is_sql_tool(tool) -> bool:
    nm = getattr(tool, "name", "") or ""
    return any(kw in nm.lower() for kw in ("adb", "sql", "oracle"))


class RunSQLInput(BaseModel):
    sql: str = Field(default="select sysdate from dual", description="The SQL query to execute.")
    model: str = Field(default="oci/generativeai-chat:2024-05-01", description="The name and version of the LLM (Large Language Model) you are using.")
    sqlcl: str = Field(default="sqlcl", description="The name or path of the SQLcl MCP client.")


def user_confirmed_tool(tool):
    async def wrapper(*args, **kwargs):
        try:
            if args and isinstance(args[0], str):
                sql_query = args[0]
                model_name = "oci/generativeai-chat:2024-05-01"
                sqlcl_param = "sqlcl"
            else:
                sql_query = kwargs.get("sql", "select sysdate from dual")
                model_name = kwargs.get("model", "oci/generativeai-chat:2024-05-01")
                sqlcl_param = kwargs.get("sqlcl", "sqlcl")

            print(f"DEBUG: Preparing payload - SQL: {sql_query}, Model: {model_name}, SQLCL: {sqlcl_param}")

            approved = True if AUTO_APPROVE == 'Y' else (
                (await asyncio.to_thread(input, "ALLOW this SQL execution? (y/n): ")).strip().lower() in {"y", "yes"}
            )

            if not approved:
                return "⚠️ Execution cancelled by user."

            payload = {"sql": sql_query, "model": model_name, "sqlcl": sqlcl_param}
            try:
                if hasattr(tool, 'ainvoke'):
                    return await tool.ainvoke(payload)
                if hasattr(tool, 'invoke'):
                    return tool.invoke(payload)
                return tool.run(**payload)
            except Exception as e:
                return f"ERROR: Failed to execute SQLcl tool - {e}\nIf 'sqlcl parameter is required', ensure SQLcl is installed and on PATH."
        except Exception as e:
            return f"ERROR: Wrapper failure: {e}"

    return StructuredTool(
        name=tool.name,
        description=getattr(tool, "description", "SQL tool"),
        args_schema=RunSQLInput,
        coroutine=wrapper,
    )


async def load_tools_from_session(name: str, session: ClientSession | None) -> list:
    """
    Load tools from a given MCP session safely.
    One bad session won't block others.
    """
    if session is None:
        return []
    try:
        tools = await load_mcp_tools(session)
        print(f"🔧 {name}: loaded {len(tools)} tools")
        return tools
    except Exception as e:
        print(f"❌ {name}: failed to load tools. Reason: {e}")
        return []


# ────────────────────────────────────────────────────────
# 4) Main
# ────────────────────────────────────────────────────────
async def main() -> None:
    async with AsyncExitStack() as stack:
         # Connect to stdio servers using the shared stack
        adb_session   = await safe_connect("Oracle SQLcl", adb_server, stack, timeout=20, retries=1)
        tavily_session= await safe_connect("Tavily", tavily_server, stack, timeout=15, retries=1)
        local_session = await safe_connect("Local File Server", local_file_server, stack, timeout=15, retries=1)

        # Connect the streamable Redis server on the same stack
        redis_session  = await safe_connect("redis", url_redis, stack, timeout=20, retries=1, http_streamable=True)

        # Connect the streamable DB Tools server on the same stack
        dbtools_session  = await safe_connect("dbtools", url_dbtools, stack, timeout=20, retries=1, http_streamable=True)


        # Now load tools as before; if redis_session is None you simply get an empty list
        all_tools = []
        all_tools += await load_tools_from_session("Oracle SQLcl", adb_session)
        all_tools += await load_tools_from_session("Tavily", tavily_session)
        all_tools += await load_tools_from_session("Local File Server", local_session)
        #all_tools += await load_tools_from_session("redis", redis_session)
        #all_tools += await load_tools_from_session("dbtools", dbtools_session)


        # Wrap SQL tools with confirmation
        final_tools = []
        for t in all_tools:
            try:
                if is_sql_tool(t):
                    final_tools.append(user_confirmed_tool(t))
                else:
                    final_tools.append(t)
            except Exception as e:
                print(f"⚠️ Skipping tool due to wrap error: {getattr(t, 'name', '<unknown>')} -> {e}")

        # Always add your local tools
        final_tools.append(run_python)
        final_tools.append(_rag_agent_service)

        print(f"✅ Registered tools: {[getattr(t, 'name', '<unnamed>') for t in final_tools]}")

        # Prompt for auto-approve
        global AUTO_APPROVE
        try:
            ans = (await asyncio.to_thread(input, "Auto-approve all SQL executions? (y/n): ")).strip().lower()
            AUTO_APPROVE = 'Y' if ans in {"y", "yes"} else 'N'
        except Exception:
            AUTO_APPROVE = 'N'

        # Build agent (works even if some servers are missing)
        agent = initialize_agent(
            tools=final_tools,
            llm=model,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True,
            agent_kwargs={"prefix": promt_oracle_db_operator},
        )


        # REPL
        history: list = []
        print("Type a question (empty / 'exit' to quit):")
        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            if not user_input or user_input.strip().lower() in {"exit", "quit"}:
                print("👋 Bye!")
                break

            history.append(HumanMessage(content=user_input))
            history = history[-30:] #### Short Term Memory

            try:
                #await agent.ainvoke({f"store this data in redis cache using hset tool : {history}", 101})
                ai_response = await agent.ainvoke({"input": history})
                
                
                # Normalize output
                eval_msg = normalize_output(ai_response, history)

                ### Log data for Eval (Long Term Memory)
                #await agent.ainvoke({"input" : f"Store a key in Redis: set hash `user:aojah1` with field {user_input} and value {eval_msg} using the hset tool. Set expiry to 7 days. Try it 1 time only."})

            except Exception as e:
                print(f"⚠️ Agent failed to respond: {e}")

# Normalize output
def normalize_output(ai_response, history):
    
    
    if isinstance(ai_response, dict):
        msg = ai_response.get("output")
    elif isinstance(ai_response, AgentFinish):
        msg = ai_response.return_values.get("output")
    else:
        msg = ai_response

    eval_msg=""
    if isinstance(msg, AIMessage):
        history.append(msg)
        eval_msg= f"AI: {msg.content}\n"
        print(eval_msg)
    elif isinstance(msg, str):
        out = AIMessage(content=msg)
        history.append(out)
        eval_msg= f"AI: {msg}\n"
        print(eval_msg)
    elif isinstance(msg, dict) and "content" in msg:
        out = AIMessage(content=msg["content"])
        history.append(out)
        eval_msg= f"AI: {out.content}\n"
        print(eval_msg)
    else:
        eval_msg= "AI: <<no response>>\n"
        print(eval_msg)

    return eval_msg

if __name__ == "__main__":
    asyncio.run(main())
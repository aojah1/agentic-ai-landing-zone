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

load_dotenv()

# ────────────────────────────────────────────────────────
# 1) Model
# ────────────────────────────────────────────────────────
model = initialize_llm()

# ────────────────────────────────────────────────────────
# 2) MCP Server Descriptors
# ────────────────────────────────────────────────────────
adb_server = StdioServerParameters(
    command=SQLCLI_MCP_PROFILE, args=["-mcp"]
)

local_file_server = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", FILE_SYSTEM_ACCESS_KEY],
)

tavily_server = StdioServerParameters(
    command="npx",
    args=["-y", "mcp-remote", TAVILY_MCP_SERVER],
)

AUTO_APPROVE = 'N'  # default, toggled at runtime


# ────────────────────────────────────────────────────────
# 3) Helpers
# ────────────────────────────────────────────────────────
async def safe_connect(name: str, params: StdioServerParameters, stack: AsyncExitStack,
                       timeout: float = 15.0, retries: int = 1) -> ClientSession | None:
    """
    Connect+initialize a MCP session safely.
    Returns ClientSession or None. Never raises.
    """
    attempt = 0
    while attempt <= retries:
        try:
            # Enter the async *context manager* returned by stdio_client(...)
            read, write = await asyncio.wait_for(
                stack.enter_async_context(stdio_client(params)),
                timeout=timeout
            )

            # Create session, then enter it as a context manager via the same stack
            session = ClientSession(read, write)
            session = await stack.enter_async_context(session)

            await asyncio.wait_for(session.initialize(), timeout=timeout)

            print(f"✅ Connected to MCP server: {name}")
            return session
        except Exception as e:
            attempt += 1
            if attempt <= retries:
                print(f"⚠️  {name}: connect failed (attempt {attempt}/{retries}). Retrying… Reason: {e}")
                await asyncio.sleep(1.5)
            else:
                print(f"❌ {name}: could not connect. Continuing without it. Reason: {e}")
                return None



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
        # Connect independently; any can fail without stopping the app
        adb_session = await safe_connect("Oracle SQLcl", adb_server, stack, timeout=20, retries=1)
        tavily_session = await safe_connect("Tavily", tavily_server, stack, timeout=15, retries=1)
        local_file_session = await safe_connect("Local File Server", local_file_server, stack, timeout=15, retries=1)

        # Load tools per session; failures are isolated
        all_tools = []
        all_tools += await load_tools_from_session("Oracle SQLcl", adb_session)
        all_tools += await load_tools_from_session("Tavily", tavily_session)
        all_tools += await load_tools_from_session("Local File Server", local_file_session)

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
            history = history[-30:]

            try:
                ai_response = await agent.ainvoke({"input": history})

                # Normalize output
                if isinstance(ai_response, dict):
                    msg = ai_response.get("output")
                elif isinstance(ai_response, AgentFinish):
                    msg = ai_response.return_values.get("output")
                else:
                    msg = ai_response

                if isinstance(msg, AIMessage):
                    history.append(msg)
                    print(f"AI: {msg.content}\n")
                elif isinstance(msg, str):
                    out = AIMessage(content=msg)
                    history.append(out)
                    print(f"AI: {msg}\n")
                elif isinstance(msg, dict) and "content" in msg:
                    out = AIMessage(content=msg["content"])
                    history.append(out)
                    print(f"AI: {out.content}\n")
                else:
                    print("AI: <<no response>>\n")
            except Exception as e:
                print(f"⚠️ Agent failed to respond: {e}")


if __name__ == "__main__":
    asyncio.run(main())
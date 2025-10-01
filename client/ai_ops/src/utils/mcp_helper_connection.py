### Helpers: safe connect + safe tool load
import asyncio
from typing import Optional, List
from langchain_core.tools import BaseTool

async def safe_connect(name: str, server_params, stack: AsyncExitStack, timeout_s: int = 8):
    """Return an initialized ClientSession or None, never raise."""
    try:
        async with asyncio.timeout(timeout_s):
            r, w = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(r, w))
            await session.initialize()
            print(f"✅ {name}: connected")
            return session
    except Exception as e:
        print(f"⚠️  {name}: unavailable ({e})")
        return None

async def safe_load_tools(name: str, session) -> List[BaseTool]:
    """Return [] on failure, never raise."""
    if session is None:
        return []
    try:
        return await load_mcp_tools(session)
    except Exception as e:
        print(f"⚠️  {name}: failed to load tools ({e})")
        return []


### Fallback “no-op” tools (so the agent never crashes)

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class TextOnlyInput(BaseModel):
    prompt: str = Field(..., description="User request or context")

def make_unavailable_tool(name: str, reason: str) -> StructuredTool:
    async def _noop(prompt: str) -> str:
        return f"{name} unavailable: {reason}. Proceeding in offline mode."
    return StructuredTool(
        name=name,
        description=f"Fallback stub for {name} when the service is down.",
        args_schema=TextOnlyInput,
        coroutine=_noop,
    )

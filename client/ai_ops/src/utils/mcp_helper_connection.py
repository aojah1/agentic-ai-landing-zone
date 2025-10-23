### Helpers: safe connect + safe tool load
import asyncio
from typing import Optional, List
from langchain_core.tools import BaseTool
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from src.utils.oci_auth_proxy import headers

async def safe_connect(name: str, params: StdioServerParameters, stack: AsyncExitStack,
                       timeout: float = 15.0, retries: int = 1, http_streamable: bool = False) -> ClientSession | None:
    """
    Connect+initialize a MCP session safely.
    Returns ClientSession or None. Never raises.
    """
    attempt = 0
    while attempt <= retries:
        try:
            
            if(http_streamable):
                # Open the stream client inside the AsyncExitStack
                ctx = streamablehttp_client(url=params)
                result = await asyncio.wait_for(
                    stack.enter_async_context(ctx),
                    timeout=timeout
                )

                # Normalize result to (read, write)
                if isinstance(result, tuple):
                    if len(result) < 2:
                        raise RuntimeError(f"streamablehttp_client returned tuple len={len(result)}; need >= 2")
                    read, write = result[0], result[1]
                else:
                    read = getattr(result, "read", None)
                    write = getattr(result, "write", None)
                    if read is None or write is None:
                        raise RuntimeError(f"Unsupported return type {type(result)}; expected tuple or object with read/write")

            else:

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



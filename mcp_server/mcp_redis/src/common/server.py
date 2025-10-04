from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse
from starlette.requests import Request
from starlette.responses import Response
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.types import Receive, Scope, Send
from starlette.routing import Mount, Route

# Initialize FastMCP server
mcp = FastMCP(
    "Redis MCP Server",
    dependencies=["redis", "dotenv", "numpy"]
)

def handle_health(request):
        return JSONResponse({"status": "success"})

mcp._additional_http_routes = [
    Route('/health', endpoint=handle_health),
]
import uvicorn
import sys
from src.common.server import mcp
import src.tools
from src.common.config import *

class OracleDBToolsMCPServer:
    def __init__(self):
        print("Starting the OracleDBToolsMCPServer", file=sys.stderr)

    # def run(self):
    #     mcp.run(transport=MCP_TRANSPORT)
    # Build an ASGI app that serves the Streamable HTTP transport.
    
    def create_app(self):
        return mcp.streamable_http_app()

    def run(self):
        
        # Run the ASGI app directly
        print(MCP_SSE_HOST)
        uvicorn.run(self.create_app(), host=MCP_SSE_HOST, port=int(MCP_SSE_PORT))

@mcp.tool()
def ping() -> str:
    """Health check."""
    return "pong"

if __name__ == "__main__":
    server = OracleDBToolsMCPServer()
    server.run()   


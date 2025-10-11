import sys
from src.common.connection import RedisConnectionManager
from src.common.server import mcp
import uvicorn
import src.tools.server_management
# import tools.misc
# import tools.redis_query_engine
import src.tools.hash
# import tools.list
# import tools.string
# import tools.json
# import tools.sorted_set
# import tools.set
# import tools.stream
# import tools.pub_sub
import src.tools.dataframe
from src.common.config import *



class RedisMCPServer:
    def __init__(self):
        print("Starting the RedisMCPServer", file=sys.stderr)

    # def run(self):
    #     mcp.run(transport=MCP_TRANSPORT)
    # Build an ASGI app that serves the Streamable HTTP transport.
    
    def create_app(self):
        return mcp.streamable_http_app()

    def run(self):
        
        # Run the ASGI app directly
        print(MCP_SSE_HOST)
        uvicorn.run(self.create_app(), host=MCP_SSE_HOST, port=int(MCP_SSE_PORT))

if __name__ == "__main__":
    server = RedisMCPServer()
    server.run()
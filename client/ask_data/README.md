ASK DATA RUN BOOK V3.0 (Date : Sep/13/2025) - ANUP OJAH

AGENT SETUP
==============

### Step 1. MCP Redis Server Setup
==================================

Deploy a Redis MCP Server following the repo here :

https://github.com/aojah1/agentic-ai-landing-zone/blob/main/mcp_server/mcp_redis/README.md

    ssh -i ssh-key-mcp-agent.key opc@192.1.1.1
    cd mcp_server/mcp_redis
    python3.13 -m venv .venv_redis_server
    source .venv_redis_server/bin/activate
    python3.13 -m pip install -r requirements.txt

    nohup python3.13 main.py > mcp_server_8003.log 2>&1 &

Note: This will start the MCP Server with http streamable protocol. 
![image.png](/client/ask_data/images/image.png)

### Step 2. MCP Agent/Client Setup
====================================

    ssh -i ssh-key-mcp-agent.key opc@192.1.1.2
    cd client/ask_data
    python3.13 -m venv .venv_agent_client
    source .venv_agent_client/bin/activate
    python3.13 -m pip install -r requirements.txt

### Run Few Test Cases to ensure MCP Client and MCP Server are able to communicate with the REDIS as a backend DB

    python3.13 -m app.getinsights.askdata_getinsights

Note: This will start an interactive session, that you can use to start interactinng with the REDIS MCP Server.
![image-3.png](/client/ask_data/images/image-3.png)

Now Go Back to the MCP Server Terminal to verify the following : 

If you see the screen below, it’s GOOD and the connection was initiated from a MCP Client for Redis cluster from the source IP/Port. —> 92.1.1.2:45012 - "POST /mcp/ HTTP/1.1" 200 OK 

![image-2.png](/client/ask_data/images/image-2.png)

Step 3. Langraph Dev Setup
=======================

    ssh -i ssh-key-mcp-agent.key opc@opc@192.1.1.2
    cd mcp_redis/mcp_server
    source .venv_agent_client/bin/activate
    nohup langgraph dev --config langgraph.json --allow-blocking --host 0.0.0.0 --port 8002 > langgraph_8002.log 2>&1 &

This is what you would expect.
![image-4.png](/client/ask_data/images/image-4.png)
You can share this URL to the consumer of the Ask Data GetInsights Agent.  The API produced will be used for down stream API to expose the LangGraph Agent to a consumer.

    http://192.1.1.2:8002/docs

Step 4. Fast API Setup (Consumer to LangGraph Agent)
============================================
    ssh -i ssh-key-mcp-agent.key opc@1opc@192.1.1.2

    cd mcp_redis/mcp_server
    source .venv_agent_client/bin/activate

    nohup python3.13 -m uvicorn app.getinsights.fastapi_getinsights:app --reload --host 0.0.0.0 --port 8004 > uvicorn_8004.log 2>&1 &

You can share this URL to the consumer of the Ask Data GetInsights Agent

    http://192.1.1.2:8004/docs 

You can now Test The System using Swagger or CURL command as described below. Also the system is now ready to be consumbed by AskData VBCS app.
![image-5.png](/client/ask_data/images/image-5.png)

#### Step 1. This is a PING Test for AskData Agent. A response 200 OK means the Agent is up and running. This url is only for testing purpose.

Get Request:

    curl -X 'GET' \
    'http://192.1.1.2:8004/askdata/search_assistant_id' \
    -H 'accept: application/json'

Expected Output: Status 200

![image-6.png](/client/ask_data/images/image-6.png)

#### Step 2: Create a Thread to be able to manage session state (Chat History). This would be ideally created during the launch of the AskData Agentic AI UI pop-up. This will ensure the entire history of conversation is maintained for that session id.

    curl -X 'GET' \
    'http://192.1.1.2:8004/askdata/getsession' \
    -H 'accept: application/json'

Expected Response : Status 200

![image-7.png](/client/ask_data/images/image-7.png)

#### Step 3: This is where you post the Prompt to the LLM and get an AI Response back

Replace the ThreadId retrieved from step 2

    curl -X 'POST' \
    'http://192.1.1.2:8004/askdata/getinsights' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "stream_mode": "values",
    "prompt": "get business insights for key 123456",
    "thread_id": "38a90807-3b42-4976-989c-3c76e76be65e"
    }'

![image-8.png](/client/ask_data/images/image-8.png)

 
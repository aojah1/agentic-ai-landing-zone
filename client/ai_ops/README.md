### =================================
## ===  DB Operator  Agent ==========
### ==================================

> The DB Operator Agent is based on the concept of Context Engineering. Here the agent is provided with a larger context to do 
it's job. Context include knowledge about local file system to work with, ability to run pythin code on a scratchpad, a RAG based knoweldge about Oracle 23.AI DB, a web search engine, OCI operation, and finally take an action against Oracle DB.

![db_operator.png](/client/ai_ops/images/db_operator.png)

# Configure your development environment

### Client Library
    cd client/ai_ops

### Configuring and running the agent
    python3.13 -m venv .venv_client_aiops
    source .venv_client_aiops/bin/activate

### Installing all the required packages

After you create a project and a virtual environment, install the latest version of required packages:
    
    python3.13 -m pip install -r requirements.txt

### Note: If your linux machine has node version < 18.0, follow the instructions below
>
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
    . ~/.nvm/nvm.sh
    nvm install --lts
    nvm use --lts
    node -v
    rm -rf ~/.npm/_npx ## remove any old references

### For SQL CL MCP Server to work, please install Java 17 or higher


### Configuring your .env (config) file
> Rename the /config/sample_.env to /config/.env
> 
> Change the config variables based on your agents requirements

### Security
The server uses OCI's built-in authentication and authorization mechanisms, including:

> OCI config file-based authentication

> Signer-based authentication for specific endpoints

# Getting started with Oracle AI Agents in 7 step :

## Step 1) System Prompt

Define the System Prompt for the Agent DB_Operator with a role in mind and following this pattern : 

CONTEXT >> ROLE >> OBJECTIVE >> FORMAT >> TONE / STYLE >> CONSTRAINTS

    python3.13 -m src.prompt_engineering.topics.db_operator

## Step 2) Configure LLM

One common place to configure access to OCI-hosted LLMs, BYO LLMs through DataScience Quick Action

#### OCI Gen AI LLM

	python3.13 -m src.llm.oci_genai

#### OCI Gen AI RAG Service

    python3.13 -m src.llm.oci_genai_agent

#### OCI Gen AI Embedding Service
		
	python3.13 -m src.llm.oci_embedding_model

#### Deploy an LLM in OCI Data Science AQUA

Deploy any LLM in OCI Data Sciane AQUA following the link below: 

    https://github.com/oracle-devrel/oci-ai-quickactions-demo

    python3.13 -m src.llm.oci_ds_md

## Step 3) Configure MCP Server

Test all of the MCP Server before building Agent Orchestration. Run the below in a Terminal, which will open a browser to that can be used to test the MCP Servers been used.

	npx @modelcontextprotocol/inspector


#### A) MCP SQLcli: 

Follow the steps per this document to configure MCP Server for SQLcli: 

    https://docs.oracle.com/en/database/oracle/sql-developer-command-line/25.2/sqcug/using-oracle-sqlcl-mcp-server.html
    https://github.com/aojah1/agentic-ai-landing-zone/blob/main/client/ai_ops/src/agents/README.md

    #Download && Install SQLcl
    mkdir sqlcl
    cd sqlcl
    wget https://download.oracle.com/otn_software/java/sqldeveloper/sqlcl-latest.zip
    unzip sqlcl-latest.zip

    #Run SQLcl and save a Connection
    ./bin/sql /nolog
    SQL> conn -save myConn -savepwd username/mypasswd@localhost:1521/freepdb1

    #Configure MCP
    {
        "mcpServers": {
            "sqlcl": {
            "timeout": 6000,
            "type": "stdio",
            "command": "/path-to/sqlcl-25.3.0/sqlcl/bin/sql",
            "args": [
                "-mcp"
            ]
            }
        }
    }

> ask questions

    "Connect to the myConn database and do X"
    
Testing:

    Transport: STDIO
    Command: /Applications/sqlcl/bin/sql
    Args: -mcp
    Prompt: select * from dual;

#### B) MCP DB_Tools: 

Follow the steps per this github repo: 

    https://github.com/aojah1/agentic-ai-landing-zone/tree/main/mcp_server/dbtools-mcp-server

Testing:

    Transport: Streamable HTTP
	Command: http://127.0.0.1:8001/mcp

#### C) MCP WebSearch with Tavily: 

Follow the steps per this website:

    https://docs.tavily.com/documentation/mcp

    Prompt Tavily Search : what is the best mexican food I can find in cancun mexico ?

Testing:

    Transport: STDIO
    Command: npx
    Args: 
        -y mcp-remote "https://mcp.tavily.com/mcp/?tavilyApiKey=xxx"



#### D) MCP LocalFile System: 

Follow the steps per this website:

    https://www.npmjs.com/package/@modelcontextprotocol/server-filesystem

Testing:

    Transport: STDIO
    Command: npx
    Args: 
        -y @modelcontextprotocol/server-filesystem /Users/...

    READ FILE:
        /Users/..../logs.txt
  
#### E) OCI RAG Service Tool: 

Uses Oracle pre-built RAG Agent Services

    python3.13 -m src.tools.rag_agent

#### E) Run Python Tool: 

Create a custom tool to run untrusted Python in a sandbox-like context.

    python3.13 -m src.tools.python_scratchpad


## Step 4) Agent Memory

DB Operator implements a short term memory using local RAM form the cpu the agent is running on

## Step 5) Agent Orchestration

The DB Operator Agent follows a Solo Agent Architecture pattern, with multiple tools.

    python3.13 -m src.agents.db_operator

## Step 6) User Interface

![alt text](/client/ai_ops/images/ui.png)

## Step 7) Agent Evaluation

![alt text](/client/ai_ops/images/metro.png)

### Build/Deploy an DB Operator Agent

> python3.13 -m src.agents.db_operator
> 
### Test DB Operator Agent

====================================

### DBA with Sys access
===================================

> Connect to Oracle DB -

    show me all connections

    use <your connections> to connect

> Verify the Installation - 

    which user I am connecting with 

    list tablespace utilization and free space. Can you plot the results in an html format and store it in my local file system. Open in a browser running open command.

> Document the Environment

    Record SID, DB name, listener ports, admin passwords (securely). Pretty print please.

========================================

### DB Developer with schema access only
=========================================

> Connect to Oracle DB -

    show me all connections

    use <your connections> to connect

    what schema I have access to 

    Use the RAG service. What are the best security principles I need to be aware of while building Vector DB in Oracle 23.ai

    Using the above best security principles, create a table test_vector_with_security_demo_1 and create a column for storing vector embedding data

    what best security principles was applied while creating this table

    get the README.md from my local file and using an embedding capability of your LLM, create a vector and store it in the table 'test_vector_with_security_demo_1' so that I can do a vector retrieval later.

    using vector retrieval capability of LLM againts test_vector_with_security_demo_1' table, what are the steps for deploying a local db ?
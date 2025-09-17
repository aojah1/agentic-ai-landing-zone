### =================================
## ===  Oracle Database Operator  ===
### ==================================

This agent integrates with Oracle DB SQLCl MCP Server, allowing NL conversation with any Oracle Database (19 c or higher).
https://docs.oracle.com/en/database/oracle/sql-developer-command-line/25.2/sqcug/using-oracle-sqlcl-mcp-server.html
Workflow Overview:
1. Load config and credentials from .env
2. Start MCP clients for SQLCL
3. Register tools with the agent
4. Run the agent with user input and print response
"""


![db_operator.png](/client/ai_ops/images/db_operator.png)


# Getting started with Oracle AI Agents in 7 step :

## Step 1) System Prompt

## Step 2) Configure LLM

## Step 3) Configure MCP Server
### A) MCP SQLcli: 

Follow the steps per this document to configure MCP Server for SQLcli: 

    https://docs.oracle.com/en/database/oracle/sql-developer-command-line/25.2/sqcug/preparing-your-environment.html

### B) MCP DB_Tools: 

Follow the steps per this github repo: 

    https://github.com/aojah1/agentic-ai-landing-zone/tree/main/mcp_server/dbtools-mcp-server

### C) MCP WebSearch with Taavily: 

Follow the steps per this website:

    https://docs.tavily.com/documentation/mcp

### D) MCP LocalFile System: 

Follow the steps per this website:
  
### E) OCI RAG Service Tool: 

Follow the steps

### E) Run Python Tool: 

Follow the steps

## Step 4) Agent Memory

## Step 5) Agent Orchestration

## Step 6) User Interface

## Step 7) Agent Evaluation

# Configure your development environment
> Fork the repository
> https://github.com/aojah1/mcp
> 
> Clone the fork locally
> 
> git clone https://github.com/<your_user_name>/mcp.git

### Optional commands
    How to actually get Python 3.13 on macOS (change it for your machine)
    Option 1 : Homebrew (simplest)
    brew update
    brew install python@3.13          # puts python3.13 in /opt/homebrew/bin
    echo 'export PATH="/opt/homebrew/opt/python@3.13/bin:$PATH"' >> ~/.zshrc
    exec $SHELL                       # reload shell so python3.13 is found
    python3.13 --version              # → Python 3.13.x
    
    Option 2 : pyenv (lets you switch versions)
    brew install pyenv
    pyenv install 3.13.0
    pyenv global 3.13.0
    python --version                  # now 3.13.0

### Client Library
    cd mcp_client

### Configuring and running the agent
    python3.13 -m venv .venv_mcp
    source .venv_mcp/bin/activate

### Installing all the required packages

After you create a project and a virtual environment, install the latest version of required packages:
> python3.13 -m pip install -r requirements.txt

### Configuring your .env (config) file
> Rename the mcp_client/config/sample_.env to mcp_client/config/.env
> 
> Change the config variables based on your agents requirements

### Security
The server uses OCI's built-in authentication and authorization mechanisms, including:

> OCI config file-based authentication

> Signer-based authentication for specific endpoints

### Build/Deploy an DB Operator Agent
This agent integrates with Oracle DB SQLCl MCP Server, allowing NL conversation with any Oracle Database (19 c or higher).

https://docs.oracle.com/en/database/oracle/sql-developer-command-line/25.2/sqcug/using-oracle-sqlcl-mcp-server.html
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

    list tablespace utilization and free space

    Verify database accessibility via sqlplus or SQL Developer

> Document the Environment

    Record SID, DB name, listener ports, admin passwords (securely)

    Capture system architecture, version details, and patch level

========================================

### DB Developer with schema access only
=========================================

    what schema I have access to 

    clear all memory

    going forward by default use the schema <<Your Schema >> every time

    create all the related tables as described in the following ERD. Use Oracle 23.AI JSON database type to create all the tables : 

    organizations
     ├── departments
     │    ├── employees
     │    │    └── roles
     │    └── budgets
     ├── goals
     │    └── objectives
     │         └── key_results
     └── projects
          ├── tasks
          └── milestones

Invoke the RAG Service with Oracle product documentation knowledge base

    tell me what security process are available in oracle 23.ai
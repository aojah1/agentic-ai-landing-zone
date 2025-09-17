### =================================
## ===  DB Operator  Agent ==========
### ==================================

> The DB Operator Agent is based on the concept of Context Engineering. Here the agent is provided with a larger context to do 
it's job. Context include knowledge about local file system to work with, ability to run pythin code on a scratchpad, a RAG based knoweldge about Oracle 23.AI DB, a web search engine, OCI operation, and finally take an action against Oracle DB.

![db_operator.png](/client/ai_ops/images/db_operator.png)


# Getting started with Oracle AI Agents in 7 step :

## Step 1) System Prompt

## Step 2) Configure LLM

## Step 3) Configure MCP Server
### A) MCP SQLcli: 

Follow the steps per this document to configure MCP Server for SQLcli: 

    https://docs.oracle.com/en/database/oracle/sql-developer-command-line/25.2/sqcug/using-oracle-sqlcl-mcp-server.html

### B) MCP DB_Tools: 

Follow the steps per this github repo: 

    https://github.com/aojah1/agentic-ai-landing-zone/tree/main/mcp_server/dbtools-mcp-server

### C) MCP WebSearch with Tavily: 

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

### Client Library
    cd mcp_client

### Configuring and running the agent
    python3.13 -m venv .venv_mcp_client_aiops
    source .venv_mcp_client_aiops/bin/activate

### Installing all the required packages

After you create a project and a virtual environment, install the latest version of required packages:
> python3.13 -m pip install -r requirements.txt

### Configuring your .env (config) file
> Rename the /config/sample_.env to /config/.env
> 
> Change the config variables based on your agents requirements

### Security
The server uses OCI's built-in authentication and authorization mechanisms, including:

> OCI config file-based authentication

> Signer-based authentication for specific endpoints

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
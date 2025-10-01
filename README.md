### Agentic AI - LandingZone
> The vision of Agetic AI LandingZone is to allow developers to build AI Agents on Oracle Cloud with speed, scalability and reliability. 
> 
> The Agentic AI LandingZone is a python project build using OCI SDK for Agent Development, Oracle GenAI and Agent Services along with few popular open-source framework like LangGraph, Langchain, FastAPI and Streamlit. The design pattern adopted allows reusability of code, good coding practice with security in mind, resulting in developers to focus more on the business logic vs spending time on the agent development engineering concepts.


## Key Concepts of how the Agentic AI Landing Zone is configured

### Applications
> An application is what gets deployed at the client side, for users or machines to interact with.
> Apps can be exposed either as an API or an UI.

### Agent Teams
> A structured sequence of steps or actions that the AI Agent follows to accomplish a specific business task or answer a user query.
Workflow patterns such as Supervisor and Swarm makes up an Agent Team.

### Agents
> Agents handles specific task and is equipped with specific skills that enables it to carry out task. Consider this as a worker behind the scenes to perform actual actions or task that the agent is suppose to deliver to the user.
Agent can connect to other systems, API's or tools, which allows the agent to utilize information from different data sources or business functions.

### Prompt Engineering

> System Prompt:
>> Each agent has a system prompt. The system prompt defines the Agents personas and capabilities. It establishes the tool it can access. It also describe how the Agent should think about achieving any goals or task for which it was designed.

Use a consistent pattern : 
CONTEXT >> ROLE >> OBJECTIVE >> FORMAT >> TONE / STYLE >> CONSTRAINTS

### llm
> One common place to configure access to OCI-hosted LLMs, BYO LLMs through DataScience Quick Action

### METRO
> MONITORING >> EVALUATION >> TRACING >> REPORTING > OBSERVABILITY

### Guardrails
> Use custom Nemo Guardrails

### Client - 

#### Configure your development environment

> Fork the repository

    https://github.com/aojah1/agentic-ai-landing-zone.git

> Clone the fork locally

    git clone https://github.com/<your_user_name>/agentic-ai-landing-zone.git

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

## Client Library
    cd client

### MCP Server - 

#### The Model Context Protocol (MCP) 

MCP is an open standard that enables developers to build secure, two‑way connections between their data sources and AI-powered tools, acting like a “USB‑C port” for AI models to access external context 

> MCP Server: 
    >> Deploy Custom functions as tools and make it available through MCP Server
    >>Follow this instruction on how to deploy your tools (Custom Functions) into Oracle DataScience using MCP architecture
https://blogs.oracle.com/ai-and-datascience/post/hosting-mcp-servers-on-oci-data-science

TBD: ADD HOW TO DEPLOY STEP BY STEP AN MCP SERVER

## MCP Server Library
    cd mcp_server

##### -- Author: Anup Ojah, HPC&AI Leader, Oracle Cloud Engineering
##### References:
https://docs.oracle.com/en-us/iaas/Content/generative-ai-agents/adk/api-reference/introduction.htm

https://www.oracle.com/applications/fusion-ai/ai-agents/

https://docs.oracle.com/en/solutions/ai-fraud-detection/index.html

https://agents.oraclecorp.com/adk/best-practices/separate-setup-run

https://agents.oraclecorp.com/adk/examples/agent-mcp-tool

https://github.com/aojah1/agents/blob/main/Agentic%20Framework_1.2_Feb03_MM_Anup.pdf


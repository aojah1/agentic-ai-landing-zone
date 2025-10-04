
# Redis MCP Server

## Overview
The Redis MCP Server is a **natural language interface** designed for agentic applications to efficiently manage and search data in Redis. It integrates seamlessly with **MCP (Model Content Protocol) clients**, enabling AI-driven workflows to interact with structured and unstructured data in Redis. Using this MCP Server, you can ask questions like:

- "Store the entire conversation in a stream"
- "Cache this item"
- "Store the session with an expiration time"
- "Index and search this vector"

## Features
- **Natural Language Queries**: Enables AI agents to query and update Redis using natural language.
- **Seamless MCP Integration**: Works with any **MCP client** for smooth communication.
- **Full Redis Support**: Handles **hashes, lists, sets, sorted sets, streams**, and more.
- **Search & Filtering**: Supports efficient data retrieval and searching in Redis.
- **Scalable & Lightweight**: Designed for **high-performance** data operations.

## Tools

This MCP Server provides tools to manage the data stored in Redis.

- `string` tools to set, get strings with expiration. Useful for storing simple configuration values, session data, or caching responses.
- `hash` tools to store field-value pairs within a single key. The hash can store vector embeddings. Useful for representing objects with multiple attributes, user profiles, or product information where fields can be accessed individually.
- `list` tools with common operations to append and pop items. Useful for queues, message brokers, or maintaining a list of most recent actions.
- `set` tools to add, remove and list set members. Useful for tracking unique values like user IDs or tags, and for performing set operations like intersection.
- `sorted set` tools to manage data for e.g. leaderboards, priority queues, or time-based analytics with score-based ordering.
- `pub/sub` functionality to publish messages to channels and subscribe to receive them. Useful for real-time notifications, chat applications, or distributing updates to multiple clients.
- `streams` tools to add, read, and delete from data streams. Useful for event sourcing, activity feeds, or sensor data logging with consumer groups support.
- `JSON` tools to store, retrieve, and manipulate JSON documents in Redis. Useful for complex nested data structures, document databases, or configuration management with path-based access.

Additional tools.

- `query engine` tools to manage vector indexes and perform vector search
- `server management` tool to retrieve information about the database

# Configure your development environment

### Server Library
    cd mcp_server/mcp_redis

### Configuring and running the agent
    python3.13 -m venv .venv_mcp_server
    source .venv_mcp_server/bin/activate

### Installing all the required packages

After you create a project and a virtual environment, install the latest version of required packages:
    
    python3.13 -m pip install -r requirements.txt

### OCI Configuration

The server requires a valid OCI config file with proper credentials. 
The default location is ~/.oci/config. For instructions on setting up this file, 
see the [OCI SDK documentation](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm).

## Configuration of Redis DB

To configure this Redis MCP Server, consider the following environment variables:

| Name                    | Description                                               | Default Value |
|-------------------------|-----------------------------------------------------------|---------------|
| `REDIS_HOST`            | Redis IP or hostname                                      | `"127.0.0.1"` |
| `REDIS_PORT`            | Redis port                                                | `6379`        |
| `REDIS_USERNAME`        | Default database username                                 | `"default"`   |
| `REDIS_PWD`             | Default database password                                 | ""            |
| `REDIS_SSL`             | Enables or disables SSL/TLS                               | `False`       |
| `REDIS_CA_PATH`         | CA certificate for verifying server                       | None          |
| `REDIS_SSL_KEYFILE`     | Client's private key file for client authentication       | None          |
| `REDIS_SSL_CERTFILE`    | Client's certificate file for client authentication       | None          |
| `REDIS_CERT_REQS`       | Whether the client should verify the server's certificate | `"required"`  |
| `REDIS_CA_CERTS`        | Path to the trusted CA certificates file                  | None          |
| `REDIS_CLUSTER_MODE`    | Enable Redis Cluster mode                                 | `False`       |

### Build/Deploy the Redis MCP Server

    python3.13 -m src.main

Note: The IP and the Port for the MCP Server is defined in the .env file 

### Test MCP Server 
Test all of the MCP Server before building Agent Orchestration. Run the below in a Terminal, which will open a browser to that can be used to test the MCP Servers been used.

	npx @modelcontextprotocol/inspector

  Testing:

    Transport: Streamable HTTP
	Command: http://127.0.0.1:8002/mcp


### Deploy the MCP Server as a Remote Hosted Server using HTTP-Streamable on OCI DataScience

    cd mcp_server/mcp_redis

    # Create a OCI Auth Token for the user name to be used
    podman login iad.ocir.io  # namespace/oracleidentitycloudservice/<user name>

podman build -t "${IMAGE}" -f Dockerfile .   # Build Docker container



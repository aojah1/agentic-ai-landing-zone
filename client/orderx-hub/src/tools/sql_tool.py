from oci.addons.adk import Agent, AgentClient
from oci.addons.adk.run.types import InlineInputLocation, ObjectStorageInputLocation
from oci.addons.adk.tool.prebuilt.agentic_sql_tool import AgenticSqlTool, SqlDialect, ModelSize
 
INLINE_DATABASE_SCHEMA = '''
                        CREATE TABLE "ADMIN"."FLIGHT_DATA"
                        (   "FLIGHT_ID" NUMBER,
                            "AIRLINE" VARCHAR2(4000 BYTE) COLLATE "USING_NLS_COMP",
                            "FROM_LOCATION" VARCHAR2(4000 BYTE) COLLATE "USING_NLS_COMP",
                            "TO_LOCATION" VARCHAR2(4000 BYTE) COLLATE "USING_NLS_COMP",
                            "Date" TIMESTAMP (6),
                            "TIME_DEPARTURE" TIMESTAMP (6),
                            "TIME_ARRIVAL" TIMESTAMP (6),
                            "PRICE" NUMBER
                        )  DEFAULT COLLATION "USING_NLS_COMP" ;
                        '''
 
INLINE_TABLE_COLUMN_DESCRIPTION = '''
                        FLIGHTS table
                        - Each row in this table represents a flight
 
                        Columns:
                        "FLIGHT_ID" - The ID of the flight
                        "AIRLINE" - The airline company of the flight
                        "FROM_LOCATION" - The location where the flight is coming from in format City, Country (AIRPORT). i.e " New York, USA (JFK)"
                        "TO_LOCATION"- The destination of the flight in format City, Country (AIRPORT). i.e " New York, USA (JFK)"
                        "Date" - the date of the flight
                        "TIME_DEPARTURE" - the time the flight departs
                        "TIME_ARRIVAL" - the time the flight arrives
                        "PRICE"- the price of the flight
                         '''
 
 
def main():
    # Use a custom agent client for custom profile and endpoints settings
    client = AgentClient(
        auth_type="api_key",
        profile="DEFAULT",
        region="us-chicago-1",
        timeout=30
    )
 
    # Instantiate a SQL Tool
    sql_tool_with_inline_schema = AgenticSqlTool(
        name="Flight SQL Tool - Inline Schema",
        description="A NL2SQL tool that retrieves flight data",
        database_schema=InlineInputLocation(content=INLINE_DATABASE_SCHEMA),
        model_size=ModelSize.LARGE,
        dialect=SqlDialect.ORACLE_SQL,
        db_tool_connection_id="ocid1.databasetoolsconnection.oc1.us-chicago-1.amaaaaaawe6j4fqaiofdxnkkcvdivnmli3kjdujchrpdaxa3p6ib3vfcuada",
        enable_sql_execution=True,
        enable_self_correction=True,
        # icl_examples=ObjectStorageInputLocation(namespace_name="namespace", bucket_name="bucket", prefix="_sql.icl_examples.txt"),
        table_and_column_description=InlineInputLocation(content=INLINE_TABLE_COLUMN_DESCRIPTION)
        # custom_instructions="instruction"
    )
 
    # Instantiate the local agent object, with SQL Tool
    agent = Agent(
        name="SQL Agent",
        client=client,
        agent_endpoint_id="ocid1.genaiagentendpoint.oc1.us-chicago-1.amaaaaaawe6j4fqannqshjszs7dd5ec45nppqvj5ljjxn7fa4iayx7cpksfa",
        tools=[sql_tool_with_inline_schema]
    )
 
    # Set up the agent once (which pushes local instructions and tools to the remote agent resource)
    agent.setup()
 
    # Run the agent with a user message
    input = "Find me flights from New York to Paris"
    response = agent.run(input)
    response.pretty_print()
 
    # Print Response Traces
    response.pretty_print_traces()
 
 
if __name__ == "__main__":
    main()
 
 
 
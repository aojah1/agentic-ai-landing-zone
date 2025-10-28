from oci.addons.adk.run.traces import RetrievalTrace, GenerationTrace, PlanningTrace, ToolInvocationTrace


def process_trace(traces: str):
    """
    # Access all traces from the response
    :param traces:
    :return:
    """

    trace_store =""
    # Iterate through each trace and process it according to its type
    for trace in traces:
        if isinstance(trace, PlanningTrace):
            input = trace.input
            output = trace.output
            usage = trace.usage
            trace_store = f"input: {input} \n output: {output} \n usage : {usage}"

        elif isinstance(trace, ToolInvocationTrace):
            tool_id = trace.tool_id
            tool_name = trace.tool_name
            invocation_details = trace.invocation_details
            trace_store = f"tool_id: {tool_id}, tool_name: {tool_name}, invocation_details : {invocation_details}"

        elif isinstance(trace, RetrievalTrace):
            input = trace.retrieval_input
            citations = trace.citations
            citation_stores = []
            for citation in citations:
                source_text = citation.source_text
                source_location_type = citation.source_location.source_location_type
                source_location_url = citation.source_location.url

                citation_store =f"source_text: {source_text}, source_location_type: {source_location_type}, source_location_url : {source_location_url}"
                citation_stores.add(citation_store)
            usage = trace.usage
            trace_store = f"input: {input}, citations: {citation_stores}, usage : {usage}"

        elif isinstance(trace, GenerationTrace):
            input = trace.input
            output = trace.generation
            usage = trace.usage
            trace_store = f"input: {input}, output: {output}, usage : {usage}"

    # Print all traces
    print("printing all traces : ")
    print(trace_store)

def test_cases():
    from src.agents.taxagent import agent_flow
    agent = agent_flow()
    agent.setup()

    # This is a context your existing code is best at producing (e.g., fetching the authenticated user id)
    client_provided_context = "[Context: The logged in user ID is: user_123] "

    # Handle the first user turn of the conversation
    input = "Get user information for user logged in."
    input = client_provided_context + " " + input
    response = agent.run(input)
    final_message = response.data["message"]["content"]["text"]
    process_trace(response.traces)


if __name__ == "__main__":
    test_cases()

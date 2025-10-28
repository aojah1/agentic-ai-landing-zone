import oci
import json
from session_demo_utils import make_security_token_signer
#Please update the CONFIG_PROFILE, agent_endpoint_id and session_id before execution the script
CONFIG_PROFILE = "DEFAULT"
agent_endpoint_id = "ocid1.genaiagentendpoint.oc1.us-chicago-1.amaaaaaax7756raadlyhnqjr44dzqm4k3h564oejuff74uvskhxsojyxsv6a"
session_id = "ocid1.genaiagentsession.oc1.us-chicago-1.amaaaaaax7756raagp62co7h53gqcbaqaeqnu7m7bfm2kkfn5ajyegph6vda"
# Service endpoint
endpoint = "https://agent-runtime.generativeai.us-chicago-1.oci.oraclecloud.com"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)
signer = make_security_token_signer(oci_config=config)
genai_agent_runtime_client = oci.generative_ai_agent_runtime.GenerativeAiAgentRuntimeClient(config=config, service_endpoint=endpoint, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240), signer=signer)
chat_details = oci.generative_ai_agent_runtime.models.chat_details.ChatDetails()
chat_details.user_message = "OCI policies for object storage"
chat_details.should_stream = False
chat_details.session_id = session_id
# use retrieve_knowledge_base_metadata.py to get the available metadata in knowledgebases
filter_conditions = {
    "filterConditions": [
        {
            "field": "topic",
            "field_type": "list_of_string",
            "operation": "contains",
            "value": ["technical", "oci"]
        },
        {
            "field": "publisher",
            "field_type": "string",
            "operation": "contains",
            "value": "maphan"
        },
        {
            "field": "year",
            "operation": "<=",
            "value": 2025,
            "field_type": "number"
        }
    ]
}
chat_details.tool_parameters = {
    "rag": json.dumps(filter_conditions)
}
print(chat_details)
chat_response = genai_agent_runtime_client.chat(agent_endpoint_id, chat_details)
# Print result
print("**************************Response of chat session **************************")
print(vars(chat_response))
if chat_details.should_stream:
    for event in chat_response.data.events():
        print(json.loads(event.data))

from agents import OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
import asyncio

def initialize_llm(model_id : str)-> OpenAIChatCompletionsModel:
    try:
        set_tracing_disabled(True)
        return OpenAIChatCompletionsModel(
            model=model_id,
            openai_client=AsyncOpenAI(
                api_key="ocigenerativeai",
                base_url="http://127.0.0.1:8088/v1/",
                max_retries=0,
            ),
            # # forwarded to chat.completions.create by most builds of agents
            # request_overrides={
            #     "temperature": 0,
            #     "response_format": {"type": "json_object"},
            #     "extra_body": {"max_completion_tokens": 800},  # translate for proxies that hate max_tokens
            # },
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise  


async def test():
    from agents import Agent, OpenAIChatCompletionsModel, Runner
    

    agent = Agent(
        name="Assistant",
        instructions="You're a helpful assistant. You provide a concise answer to the user's question.",
        model=initialize_llm("xai.grok-3"),
    )

    result = await Runner.run(agent, "Tell me about recursion in programming.")
    print(result.final_output)

if __name__ == "__main__":
     asyncio.run(test())


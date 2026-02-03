import asyncio
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from agents import Agent, Runner, function_tool, trace


@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."


agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)


async def main():
    with trace('Agent function call'):
        result = await Runner.run(agent, input="What's the weather in Tokyo?")
        print(result.final_output)
        # The weather in Tokyo is sunny.


if __name__ == "__main__":
    asyncio.run(main())
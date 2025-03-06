import asyncio

from browser_use import Agent
from dotenv import load_dotenv

from examples.factory.ai_factory import create_chat_ai

load_dotenv()


async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=create_chat_ai(),
    )
    await agent.run()


asyncio.run(main())

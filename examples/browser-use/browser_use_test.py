import asyncio

from browser_use import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=ChatOpenAI(model="gpt-4o"),
    )
    await agent.run()


asyncio.run(main())

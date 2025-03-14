import asyncio

from browser_use import Agent
from dotenv import load_dotenv

from examples.factory.ai_factory import create_chat_ai
from examples.factory.llm import LLMFactory, LLMType

load_dotenv()

llm_factory = LLMFactory(
    llm_type=LLMType.LLM_TYPE_QWENAI,
)

async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=llm_factory.create_llm(),
    )
    await agent.run()


asyncio.run(main())

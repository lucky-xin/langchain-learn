import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
from langgraph.prebuilt import create_react_agent

from examples.factory.llm import LLMFactory, LLMType


async def run(q: str) -> None:
    llm_factory = LLMFactory(
        llm_type=LLMType.LLM_TYPE_QWENAI,
    )
    async with MultiServerMCPClient(
            {"searcher": SSEConnection(url="http://localhost:8888/sse", transport="sse")}
    ) as client:
        agent = create_react_agent(llm_factory.create_llm(), client.get_tools())
        resp = await agent.ainvoke({"messages": q})
        messages = resp.get("messages", [])
        for message in messages:
            print(message.content)


def main() -> None:
    q = '帮我介绍圣城耶路撒冷'
    asyncio.run(run(q))


if __name__ == '__main__':
    main()

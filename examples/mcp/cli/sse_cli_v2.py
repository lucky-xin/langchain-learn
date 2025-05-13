import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


async def run(q: str) -> None:
    llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model='qwq-plus-latest',
        temperature=0.2,
        streaming=True,
    )
    async with MultiServerMCPClient(
            {
                "searcher": {
                    # make sure you start your weather server on port 8888
                    "url": "http://localhost:8888/sse",
                    "transport": "sse",
                }
            }
    ) as client:
        agent = create_react_agent(llm, client.get_tools())
        resp = await agent.ainvoke({"messages": q})
        messages = resp.get("messages", [])
        for message in messages:
            print(message.content)


def main() -> None:
    q = '帮我介绍圣城耶路撒冷'
    asyncio.run(run(q))


if __name__ == '__main__':
    main()

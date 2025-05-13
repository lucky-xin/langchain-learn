import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    async with sse_client("http://localhost:8888/sse") as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            # List available tools
            tools = await session.list_tools()
            # Call a tool
            result = await session.call_tool(
                name='searcher',
                arguments={
                    "keyword": "Tomorrow's weather in SF?",
                    "page_num": 1
                }
            )

    print(tools.tools)
    print('----------')
    print(result.content)
    return session


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import os

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from examples.factory.llm import LLMFactory, LLMType

llm_factory = LLMFactory(llm_type=LLMType.LLM_TYPE_QWENAI)
llm = llm_factory.create_llm()
environ = os.environ.copy()
venv_site_packages = '/opt/homebrew/lib/python3.12/site-packages'
if 'PYTHONPATH' in environ:
    environ['PYTHONPATH'] = f"{venv_site_packages}:{environ['PYTHONPATH']}"
else:
    environ['PYTHONPATH'] = venv_site_packages
server_params = StdioServerParameters(
    command="python3",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["/langchain-learn/examples/mcp/math/math_server.py"],
    env=environ
)


async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
    # Get tools
    tools = await load_mcp_tools(session)
    # Create and run the agent
    agent = create_react_agent(llm, tools)
    agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
    return agent_response


# Run the async function
if __name__ == "__main__":
    result = asyncio.run(run_agent())
    print(result)

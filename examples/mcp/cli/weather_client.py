from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(model="gpt-4o")
with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["/path/to/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                # make sure you start your weather server on port 8000
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
) as client:
    agent = create_react_agent(model, client.get_tools())
    math_response = agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
    print(math_response)
    weather_response = agent.ainvoke({"messages": "what is the weather in nyc?"})
    print(weather_response)

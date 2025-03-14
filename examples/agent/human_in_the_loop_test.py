from typing import Annotated

from langchain import hub
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

from examples.factory.llm import LLMFactory, LLMType


class State(TypedDict):
    messages: Annotated[list, add_messages]




@tool
def chatbot(state: State):
    """Chatbot is thinking"""
    message = agent_executor.invoke({"messages": state["messages"]})
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    resp = interrupt({"query": query})
    print(f"resp:{resp}")
    return resp["data"]

class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    """Chatbot is thinking"""
    message = agent_executor.invoke({"messages": state["messages"]})
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


tools = [
    chatbot,
    human_assistance
]

prompt = hub.pull("wfh/react-agent-executor")
prompt.pretty_print()
llm_factory = LLMFactory(
    llm_type=LLMType.LLM_TYPE_QWENAI,
)
agent_executor = create_react_agent(
    model=llm_factory.create_llm(),
    tools=tools,
    messages_modifier=prompt,
    version="v2",
)

graph_builder = StateGraph(State)
tool_node = ToolNode(tools=tools)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)

from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

human_command = Command(resume={"data": human_response})
config = {"configurable": {"thread_id": "1"}}
events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, END, StateGraph
from langgraph.prebuilt import ToolNode

from examples.factory.ai_factory import create_ai


@tool
def search_online(query: str):
    """搜索天气"""
    if "上海" in query.lower() or "Shanghai" in query.lower():
        return "上海"
    return "北京"


def should_continue(state: MessagesState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return END


tools = [search_online]
tool_node = ToolNode(tools)
llm = create_ai()


def call_llm(state: MessagesState):
    messages = state["messages"]
    resp = llm.invoke(messages)
    return {"messages": [resp]}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_llm)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue
)

workflow.add_edge("tools", "agent")

ms = MemorySaver()
app = workflow.compile(checkpointer=ms)

ui = {"messages": [HumanMessage(content="北京天气怎么样？")]}

final_state = app.invoke(input=ui, config={"configurable": {"thread_id": 42}})

result = final_state["messages"][-1].content

graph_png = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_png)

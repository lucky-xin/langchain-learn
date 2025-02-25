import functools
import operator
from typing import Annotated, Literal, TypedDict, Sequence

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, BaseTool
from langchain_experimental.utilities import PythonREPL
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from examples.factory.ai_factory import create_ai


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


def agent_node(state, agent, name):
    # 调用代型
    result = agent.invoke(state)
    # 检查 result 是否是 ToolMessage 类型的实例
    if isinstance(result, ToolMessage):
        pass
    else:  # 将 tavily result 转换为 AIMessage 类型，并且将 name 作为发送者的名称附加到消息中
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # 由于我们有一个严格的工作流程，我们可以
        # 跟踪发送者，以便知道下一个传递给准。
        "sender": name,
    }


def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # 这是路山器
    messages = state["messages"]
    last_message = messages[-1]
    # 检査 last_message 是否包含工具调用(tool calls)
    if last_message.tool_calls:
        return "call_tool"
    # 如果已经获取到最终答案，则返回结束节点
    if "FINAL ANSWER" in last_message.content:
        # 任何代型决定工作完成
        return END
    return "continue"


def create_agent(llm: BaseChatOpenAI, ts: [BaseTool], system_message: str):
    """
    创建一个代理，它使用指定的LLM和工具集来执行任务。
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                 你是一个有帮助的AI助手，与其他助手合作。使用提供的工具来推进问题的回答。
                 如果你不能完全回答，没关系，另一个拥有不同工具的助手会接着你的位置继续帮助。执行你能做的以取得进展。
                 如果你或其他功手有最终答案或交付物，在你的回答前加上FINAL ANSWER，以便团队知道停止。
                 你可以使用以下工具:{tool_names}。\n{system_message}
                 """
            )
            ,
            # 消息占位符
            MessagesPlaceholder(variable_name="messages"),
        ])
    prompt = prompt.partial(system_message=system_message)
    return prompt | llm.bind(tools=ts)


repl = PythonREPL()


@tool
def python_repl(code: Annotated[str, "要执行以生成图表的Python代码"]):
    """
    使用这个工员来执行Python代码。如果你想查看某个值的输出应该使用print(...)。这个输出对用户可见。
    """
    try:
        result = repl.run(code)
    except Exception as e:
        return f"Error: {str(e)}"
    # 返回执行结果
    result_str = f"成功执行:\n'`python\n{code}\n`'\nStdout: {result}"
    return (
            result_str + "\n\n如果你已完成所有任务，请回复FINAL ANSWER。"
    )


tools = [DuckDuckGoSearchResults(), python_repl]
tool_node = ToolNode(tools)
chat_llm = create_ai()

research_agent = create_agent(
    chat_llm,
    ts=[
        DuckDuckGoSearchResults()
    ],
    system_message="你应该提供准确的数据供chart_generator使用"
)

research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# 图表生成器
chart_agent = create_agent(
    chat_llm,
    [python_repl],
    system_message="你展示的仟何图表都对用户可见。"
)
# 创建图表生成节点
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")
# 导入预构建的工具节点
# 创建状态图实例
workflow = StateGraph(AgentState)
# 添加酬究员节点workflow.add_node("Researcher",research_node)#添加图表生成器节点
workflow.add_node("chart_generator", chart_node)
# 添加工具调用节点
workflow.add_node("call_tool", tool_node)

# 添加条竹边
workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END}
)

workflow.add_conditional_edges(
    "chart_generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "__end__": END}
)

# 添加条件边
workflow.add_conditional_edges(
    "call_tool",
    # 这个 lambda 雨数的作用是从状态中获取sender名称，以便在条件边的映射中使用
    # 如果 sender冠"Researcher"，工作流将转移到"Researcher” 节点。
    # 如果 sender 是"chart_generator"，工作流将转移到"chart_generator”节点
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "chart_generator": "chart_generator"}
)

# 添加起始边
workflow.add_edge(START, "Researcher")
# 译工作流图
graph = workflow.compile()
# 将生成的图片保存到文件
graph_png = graph.get_graph().draw_mermaid_png()
with open("collaboration.png", "wb") as f:
    f.write(graph_png)

# 事件流
events = graph.stream(
    input={
        "messages": [
            HumanMessage(
                content="获取过去5年AI软件市场规模，然后绘制一条折线图。 一旦你编写好代码，完成任务。"
            )]
    },
    # 图小最多执行的步骤数
    config={"recursion_limit": 150}
)

for e in events:
    print(e)
    print("---")

import uuid
from typing import List, Literal

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.messages import SystemMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import END, MessageGraph
from pydantic import BaseModel

from examples.factory.ai_factory import create_ai

prompt_text = """
你的工作是从用户那里获取他们想要创建哪种类型的提示模板的信，您应该从他们那里获得以下信息:
-提示的目的是什么
-将向提示模板传递哪些变量
-输出不应该做什么的任何限制
-输出必须遵守的任何要求
如果你无法辨别这些信息，请他们澄清!不要试图疯狂猜测。在您能够辨别所有信息后，调用相关工具
"""

# 定义一个新的系统提示模板
prompt_system = """
Based on the following requirements, write a good prompt template:
{reqs}
"""


# 定义一个数据模型，用于存储提示模板的指令信息
class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""
    objective: str
    variables: List[str]
    constraints: List[str]
    requirements: List[str]


def get_messages_info(messages: [BaseMessage]):
    return [SystemMessage(content=prompt_text)] + messages


# 定义一个函数，用于获取生成提示模板所荒的消息，只获取工具调用之后的消息
def get_prompt_messages(messages: [BaseMessage]):
    reqs = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            reqs = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif reqs is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=reqs))] + other_msgs


# 定义一个函数，用于获取当前状态
def get_state(messages: [BaseMessage]) -> Literal["add_tool_message", "info", "__end__"]:
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"


workflow = MessageGraph()


@workflow.add_node
def add_tool_message(state: list):
    return ToolMessage(
        content="Prompt generated!", tool_call_id=state[-1].tool_calls[0]["id"]
    )


llm = create_ai().bind_tools([PromptInstructions])

chain = get_messages_info | llm

# 将消息处理链定义为 get_prompt_messages 函数和 LLM 实例
prompt_gen_chain = get_prompt_messages | llm

# 初始化 MemorySaver 共例
memory = MemorySaver()

workflow.add_node("info", chain)
workflow.add_node("prompt", prompt_gen_chain)

workflow.add_conditional_edges("info", get_state)

workflow.add_edge(START, "info")
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)

graph = workflow.compile(checkpointer=memory)

graph_png = graph.get_graph().draw_mermaid_png()
with open("prompt_gen_agent.png", "wb") as f:
    f.write(graph_png)

config = {"configurable": {"thread_id": str(uuid.uuid4()), "recursion_limit": 50}}
# 目的:['收集客户满意度反馈']
# 变量:['客户名称'、"瓦动日期'、"提供的服务'、'评级(1-5 级)'、'评论']
# 约束:["输出不应包含客户的任何个人身份信息(PII)。']
# 要求:['输出必须包含结构化格式，其中包含上述每个变量的字段。']
while True:
    q = input("请输入用户信息:")
    if q in {'Q', 'q'}:
        print("退出程序")
        break
    output = None
    for output in graph.stream(
            [HumanMessage(content=q)],
            config=config,
            stream_mode="updates",
    ):
        last_message = next(iter(output.values()))
        last_message.print()

        if output and "prompt" in output:
            print("prompt:", output["prompt"].content)

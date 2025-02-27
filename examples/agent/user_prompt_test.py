import inspect
import threading
import uuid
from typing import Callable, TypeVar
from typing import List, Annotated

import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, BaseMessage, AIMessageChunk
from langchain_core.messages import SystemMessage
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langgraph.constants import START
from langgraph.graph import END, add_messages, StateGraph
from pydantic import BaseModel, Field
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from typing_extensions import TypedDict

from examples.agent.postgres_saver_factory import create_checkpointer
from examples.factory.ai_factory import create_ai


def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    fn_return_type = TypeVar('fn_return_type')

    def add_streamlit_context(fn: Callable[..., fn_return_type]) -> Callable[..., fn_return_type]:
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> fn_return_type:
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)

        return wrapper

    st_cb = StreamlitCallbackHandler(
        parent_container=parent_container,
        collapse_completed_thoughts=False,
    )

    for method_name, method_func in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if method_name.startswith('on_'):
            setattr(st_cb, method_name, add_streamlit_context(method_func))
    return st_cb


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# 定义一个数据模型，用于存储提示模板的指令信息
class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""
    objective: str = Field(description="目标")
    variables: List[str] = Field(description="变量列表")
    constraints: List[str] = Field(description="约束列表")
    requirements: List[str] = Field(description="要求列表")


# 定义一个新的系统提示模板
prompt_system = """
Based on the following requirements, write a good prompt template:
{reqs}
"""


@st.cache_resource(ttl="1d")
def create_prompt() -> BasePromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                你的工作是从用户那里获取他们想要创建哪种类型的提示模板的信，您应该从他们那里获得以下信息:
                -提示的目的是什么
                -将向提示模板传递哪些变量
                -输出不应该做什么的任何限制
                -输出必须遵守的任何要求
                如果你无法辨别这些信息，请他们澄清!不要试图疯狂猜测。在您能够辨别所有信息后，并调用相关工具
                """
            ),
            ("placeholder", "{messages}"),
        ]
    )

# 定义一个函数，用于获取生成提示模板所荒的消息，只获取工具调用之后的消息
def get_prompt_messages(state: State):
    reqs = None
    other_msgs = []
    for m in state["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            reqs = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif reqs is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=reqs))] + other_msgs


# 定义一个函数，用于获取当前状态
def get_state(state: State) -> str:
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"


def get_info_chain(state: State):
    messages = state["messages"]
    chain = create_prompt() | llm_with_tool
    resp = chain.invoke({"messages": messages})
    return {"messages": [resp]}


def prompt_gen_chain(state: State):
    # 将消息处理链定义为 get_prompt_messages 函数和 LLM 实例
    messages = get_prompt_messages(state)
    resp = llm.invoke(messages)
    return {"messages": [resp]}


def tool_chain(state: State):
    return {
        "messages": [
            ToolMessage(
                content="Prompt generated!",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }


def init():
    print("init...")
    # Process any pending human intervention first
    if "human_response" not in st.session_state:
        st.session_state.human_response = None
    if "waiting_for_human" not in st.session_state:
        st.session_state.waiting_for_human = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


llm = create_ai()
llm_with_tool = llm.bind_tools([PromptInstructions])

# Create graph builder
if not st.session_state.get("graph"):
    print("Creating graph...")
    # 初始化 MemorySaver 共例
    workflow = StateGraph(State)
    workflow.add_node("info", get_info_chain)
    workflow.add_node("prompt", prompt_gen_chain)
    workflow.add_node("add_tool_message", tool_chain)

    workflow.add_conditional_edges("info", get_state, ["add_tool_message", "info", END])

    workflow.add_edge(START, "info")
    workflow.add_edge("add_tool_message", "prompt")
    workflow.add_edge("prompt", END)
    checkpointer = create_checkpointer()
    graph = workflow.compile(checkpointer=checkpointer)
    st.session_state.graph = graph
    st.session_state.run_id = str(uuid.uuid4())

st.image(
    image=st.session_state.graph.get_graph().draw_mermaid_png(),
    caption="LangGraph Visualization",
    use_container_width=False
)


def invoke(user_input: str):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    collected_messages = ""
    with st.chat_message("assistant"):
        st_cb = get_streamlit_cb(st.container())
        config = {
            "configurable": {
                "run_id": st.session_state.run_id,
                "recursion_limit": 50,
                "thread_id": str(threading.current_thread().ident)
            },
            "callbacks": [st_cb]
        }
        output_placeholder = st.empty()
        for chunk in st.session_state.graph.stream(
                input={"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
        ):
            first_mc = chunk[0]
            if isinstance(first_mc, AIMessageChunk) and first_mc.content:
                collected_messages += first_mc.content
                # output_placeholder.markdown(collected_messages + "▌")
        # output_placeholder.markdown(collected_messages)
        st.session_state.messages.append({"role": "assistant", "content": collected_messages})

# 目的:['收集客户满意度反馈']
# 变量:['客户名称'、"瓦动日期'、"提供的服务'、'评级(1-5 级)'、'评论']
# 约束:["输出不应包含客户的任何个人身份信息(PII)。']
# 要求:['输出必须包含结构化格式，其中包含上述每个变量的字段。']
init()
if q := st.chat_input("请输入需求信息"):
    invoke(q)

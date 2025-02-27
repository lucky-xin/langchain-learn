import asyncio
from typing import List, TypedDict

import streamlit as st
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchResults, TavilySearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from examples.factory.ai_factory import create_ai


class PPT(BaseModel):
    """The generate result of PPT"""
    title: str = Field(..., description="PPT title")
    contents: List[str] = Field(..., description="PPT contents")


class EditResp(BaseModel):
    """The edit result of PPT"""
    title: str = Field(..., description="PPT title")
    contents: List[str] = Field(..., description="PPT contents")
    canceled: bool = Field(..., description="If the user wants to cancel the generation")
    ask: str = Field(..., description="More information is required from the user")


class HumanResp(BaseModel):
    """The response of human"""
    requirements: str = Field(..., description="user's feedback")


class GenState(TypedDict):
    """The state of PPT generate"""
    topic: str
    role: str
    title: str
    requirements: str
    contents: List[str]
    canceled: bool
    ask: str

tools = [
    WikipediaQueryRun(
        name="wiki-tool",
        description="look up things in wikipedia",
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100),
    ),
    DuckDuckGoSearchResults(),
    TavilySearchResults(),
]

gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            你的目标是作为{role}基于主题{topic}编写PPT，为了更好的完成PPT写作，你可以用工具搜索你需要的内容。
            
            用户要求如下：
            {requirements}
            
            你需要以JSON结构返回。JSON内容包括以下字段： 
                title：PPT 标题；
                contents：PPT 内容，为字符串数组，每一张PPT为数组一个元素，以markdown格式输出；

            请基于用户需求写PPT，如果用户提供的信息不足以写PPT，则给予“请提供更多信息，好让我帮你写PPT!”
            """
        ),
        ("placeholder", "{messages}"),
    ]
)

gen_llm = create_ai().with_structured_output(schema=PPT)
generate_agent = gen_prompt | gen_llm

prompt = hub.pull("wfh/react-agent-executor")
prompt.pretty_print()

edit_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            你的目标是作为{role}基于主题{topic}编写PPT，目前你已经已经把PPT内容返回给用户，用户将根据他自身喜好，对PPT内容进行修改，你需要根据用户反馈进行下一步工作。
            
            用户的反馈是：
            {requirements}
            
            基于用户的反馈，你需要判断用户意图以及工作如下：
                1.根据用户反馈信息对PPT进行修改；
                2.如果用户想重新生成PPT则重新生成PPT；
                3.用户想取消；
                4.如果不明白用户意图则向用户提问；
                
            你需要以JSON结构返回。JSON内容包括以下字段： 
                title：PPT 标题；
                contents：PPT 内容，为字符串数组，每一张PPT为数组一个元素，以markdown格式输出；
            """
        ),
    ])

edit_agent = edit_prompt | create_ai().with_structured_output(schema=EditResp)


async def create_ppt_gen_graph() -> StateGraph:
    generator_key = "generator"
    user_key = "user"
    editor_key = "editor"

    async def generator_action(state: GenState) -> PPT:
        resp: PPT = await generate_agent.ainvoke(state)
        return resp

    async def editor_action(state: GenState) -> EditResp:
        agent_response: EditResp = await edit_agent.ainvoke(state)
        return agent_response

    async def human_action(state: GenState) -> HumanResp:
        ask = state.get("ask")
        if ask:
            st.chat_message("assistant").markdown(ask)

        return None

    async def dispatch(state: GenState) -> str:
        require = state.get("require", None)
        selected = state.get("selected", None)
        canceled = state.get("canceled", False)

        if canceled:
            return END
        if require and selected:
            return editor_key

        if not require and not selected:
            return user_key

        return generator_key

    workflow = StateGraph(GenState)

    workflow.add_node(generator_key, generator_action)
    workflow.add_node(user_key, human_action)
    workflow.add_node(editor_key, editor_action)

    workflow.add_edge(START, generator_key)
    workflow.add_edge(generator_key, user_key)
    workflow.add_edge(editor_key, user_key)

    checkpointer = InMemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    graph_png = app.get_graph().draw_mermaid_png()
    with open("agent_workflow.png", "wb") as f:
        f.write(graph_png)

    config = {
        "configurable": {"thread_id": "thread-1"},
        "recursion_limit": 50
    }
    inputs = {
        "topic": "人工智能",
        "num": 3,
        "role": "IT架构师",
        "require": "吸引人眼球",
    }

    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != END:
                print(f"{k}: {v}")
    return workflow


asyncio.run(create_ppt_gen_graph())

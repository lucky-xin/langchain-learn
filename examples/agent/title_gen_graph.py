import asyncio
import operator
from typing import List, TypedDict, Annotated, Tuple

from langchain import hub
from langchain_community.tools import DuckDuckGoSearchResults, TavilySearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from examples.factory.ai_factory import create_ai


class PPTGenReq(BaseModel):
    """The generator require of PPT"""
    role: str = Field(..., description="The role of writer PPT")
    topic: str = Field(..., description="The topic of PPT")
    titles: int = Field(default=3, description="The total titles of PPT")
    chapters: int = Field(..., description="The total chapters of PPT")

    def prompt_chapter(self, title: str, require: str):
        return f"""
        You are a professional PPT chart generator.
        You are given the following information:
        - Topic: {self.topic}
        - Number of chapters: {self.num}
        - Role of writer: {self.role}

        I hope you can generate an outline with only headings based on {title} using Markdown format, and please adhere to the following requirements:
        If you want to create a heading, add a hash symbol (#) before the word or phrase. The number of # symbols represents the level of the heading.
        Do not use unordered or ordered lists; you must use the hash symbol (#) method to represent the outline structure entirely.
        The first level (#) represents the outline's title, the second level (##) represents the chapter's title, and the third level (###) represents the key points of the chapter.
        For the outline, the requirement is follow:
            {require}
        
        The first chapter of the outline is an introduction to {title}, and the last chapter is a summary.
        Please generate {self.num} chapters for the PPT.
        """

    def prompt_content(self, require: str):
        return f"""
        You are a professional PPT content generator.
        You are given the following information:
        - Topic: {self.topic}
        - Title: {self.title}
        - Role of writer: {self.role}
        - Requirements: 
            {require}
        
        Please generate content for the PPT.
        """


class TitleEditReq(BaseModel):
    """The edit require of PPT title"""
    topic: str = Field(..., description="The topic of PPT")
    role: str = Field(..., description="The role of writer PPT")
    title: str = Field(..., description="The original PPT title")
    require: str = Field(..., description="The edit require of PPT title")


class ChapterEditReq(BaseModel):
    """The edit require of PPT chapter"""
    topic: str = Field(..., description="The topic of PPT")
    title: str = Field(..., description="The title of PPT ")
    chapter: str = Field(..., description="The original PPT chapter")
    require: str = Field(..., description="The edit require of PPT chapter")
    role: str = Field(..., description="The role of writer PPT")

    def prompt(self):
        return f"""
        You are a professional PPT chapter editor.
        You are given the following information:
        - Topic: {self.topic}
        - Role of writer: {self.role}
        - Original title: {self.title}
        - Original chapter: {self.chapter}
        - Requirements: 
            {self.require}
        
        Please revise the chapter based on the user's feedback.
        """


class ContentEditReq(BaseModel):
    """The edit require of PPT content"""
    topic: str = Field(..., description="The topic of PPT")
    title: str = Field(..., description="The title of PPT ")
    chapter: str = Field(..., description="The chapter of PPT")
    require: str = Field(..., description="The edit require of PPT content")
    content: str = Field(..., description="The original PPT content")
    role: str = Field(..., description="The role of writer PPT")

class TitleGenResp(BaseModel):
    """The generate result of PPT title"""
    titles: List[str] = Field(..., description="PPT titles")


class TitleEditResp(BaseModel):
    """The edit result of PPT title"""
    title: str = Field(..., description="PPT title")


class ChapterGenResp(BaseModel):
    """The generate result of PPT chapter"""
    contents: List[str] = Field(..., description="PPT chapters")


class ContentGenResp(BaseModel):
    """The generate result of PPT content"""
    content: str = Field(..., description="PPT content")


class ArbitrateResp(BaseModel):
    """The response of arbitrator"""
    selected: str = Field(..., description="The title the user has chosen to modify")
    require: str = Field(..., description="The require of PPT title")
    ask: str = Field(..., description="More information is required from the user")


class HumanResp(BaseModel):
    """The response of human"""
    require: str = Field(..., description="user's feedback")


class TitleGenState(TypedDict):
    """The state of title generate"""
    topic: str
    role: str
    require: str
    num: int
    selected: str
    canceled: bool
    titles: Annotated[List[Tuple], operator.add]
    ask: str


class WikiInputs(BaseModel):
    """Inputs to the wikipedia tool."""
    query: str = Field(description="query to look up in Wikipedia, should be 3 or less words")


tools = [
    WikipediaQueryRun(
        name="wiki-tool",
        description="look up things in wikipedia",
        args_schema=WikiInputs,
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100),
    ),
    DuckDuckGoSearchResults(),
    TavilySearchResults(),
]

ppt_gen_req = PPTGenReq(role="IT架构师", topic="给公司内部人员科普大模型", titles=3, chapters=4)

gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a professional PPT writer.
            You are given the following information:
            - Topic: {topic}
            - Number of titles: {num}
            - Role of writer: {role}
            - Requirements: 
                {require}
            
            Please generate {num} titles for the PPT.
            """
        ),
        ("placeholder", "{messages}"),
    ]
)

gen_llm = create_ai().with_structured_output(schema=TitleGenResp)
generate_agent = gen_prompt | gen_llm

prompt = hub.pull("wfh/react-agent-executor")
prompt.pretty_print()

edit_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            你的目标是修改PPT标题，用户已经给你了标题，你需要根据用户反馈进行修改。
            
            PPT主题是：
            {topic}
            
            PPT编写角色是：
            {role} 
                        
            你目前已给用户生成的标题是:
            {selected}
            
            用户的反馈是：
            {require}
            
            你需要根据用户的反馈，修改标题和计划。
            """
        ),
    ])

edit_agent = edit_prompt | create_ai().with_structured_output(schema=TitleEditResp)

arbitrate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            你的目标是生成{num}个备选的PPT标题，目前你已经已经把PPT标题返回给用户，用户根据他自身喜好，对标题进行挑选和修改，你需要根据用户反馈进行下一步工作。
            
            PPT主题是：
            {topic}
            
            PPT编写角色是：
            {role} 
                        
            你目前已给用户生成的标题列表是:
            {titles}
            
            用户的反馈是：
            {require}
            
            对于给定的PPT标题和用户反馈。你需要判断用户意图如下：
                1.用户选择了哪个标题来做最后的PPT标题
                2.用户选定某一个标题来进行修改
                3.用户想要重新生成所有备选标题
                4.用户想取消
            
            你需要以JSON结构返回。JSON结构体包含selected、require、ask、canceled。
            selected字段是用户不需要进行任何修改时选定了最终的标题或者是用户想要用来进行修改的标题。require字段是用户修改意见或者重新创建所有备选标题的建议。ask字段是不明白用户意图时，需要向用户提问的问题。如果用户想取消则canceled为true。
            """
        ),
        ("placeholder", "{messages}"),
    ]
)

arbitrate_llm = create_ai().with_structured_output(schema=ArbitrateResp)
arbitrate_agent = arbitrate_prompt | arbitrate_llm

async def create_title_gen_state_graph() -> StateGraph:
    user_key = "user"
    arbitrator_key = "arbitrator"
    generator_key = "generator"
    editor_key = "editor"

    async def generator_action(state: TitleGenState) -> TitleGenResp:
        resp: TitleGenResp = await generate_agent.ainvoke(state)
        return resp

    async def editor_action(state: TitleGenState) -> TitleEditResp:
        agent_response: TitleEditResp = await edit_agent.ainvoke(state)
        return agent_response

    async def arbitrator_action(state: TitleGenState) -> ArbitrateResp:
        resp: ArbitrateResp = await arbitrate_agent.ainvoke(state)
        return resp

    async def human_action(state: TitleGenState) -> HumanResp:
        resp: HumanResp = await arbitrate_agent.ainvoke(state)
        return resp

    async def to_user(state: TitleGenState) -> str:
        if state.get("canceled"):
            return END
        return user_key

    async def to_arbitrator(state: TitleGenState) -> str:
        if state.get("canceled", False):
            return END
        return arbitrator_key

    async def dispatch(state: TitleGenState) -> str:
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

    workflow = StateGraph(TitleGenState)

    workflow.add_node(generator_key, generator_action)
    workflow.add_node(user_key, human_action)
    workflow.add_node(editor_key, editor_action)
    workflow.add_node(arbitrator_key, arbitrator_action)

    workflow.add_edge(START, generator_key)
    workflow.add_edge(generator_key, user_key)
    workflow.add_edge(editor_key, user_key)
    workflow.add_edge(user_key, arbitrator_key)

    workflow.add_edge(arbitrator_key, editor_key)
    workflow.add_edge(arbitrator_key, generator_key)
    workflow.add_edge(arbitrator_key, user_key)

    workflow.add_conditional_edges(generator_key, to_user)
    workflow.add_conditional_edges(editor_key, to_user)
    workflow.add_conditional_edges(user_key, to_arbitrator)
    workflow.add_conditional_edges(arbitrator_key, dispatch)

    checkpointer = InMemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    graph_png = app.get_graph().draw_mermaid_png()
    with open("agent_workflow.png", "wb") as f:
        f.write(graph_png)

    config = {"configurable": {"thread_id": "thread-1", "recursion_limit": 50}}
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


asyncio.run(create_title_gen_state_graph())

import asyncio
import operator
from typing import Annotated, List, Tuple, TypedDict, Union

from langchain import hub
from langchain_community.chat_models import ChatTongyi
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """用户计划"""
    steps: List[str] = Field(
        description="需要执行的不同步骤，应该按照顺序排列"
    )


class Response(BaseModel):
    """用户响应"""
    response: str


class Action(BaseModel):
    """需要执行的行为"""
    action: Union[Response, Plan] = Field(
        description="要执行的行为。如果要回应用户，使用Response。如果需要进一步使用工具获取答案，使用Plan"
    )


tools = [TavilySearchResults(max_results=1)]

prompt = hub.pull("wfh/react-agent-executor")
prompt.pretty_print()

llm = ChatTongyi(max_retries=5)

agent_executor = create_react_agent(llm, tools, messages_modifier=prompt)

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            对于给定的目标，提出一个简单的逐步计划。这个计划应该包含独立的任务，如果正确执行将得出正确的答案。
            不要添加任何多余的步骤。最后一步的结果应该是最终答案。
            确保每一步都有所有必要的信息 -不要跳过步骤。
            """
        ),
        ("placeholder", "{messages}"),
    ]
)

planner_llm = ChatTongyi(max_retries=5).with_structured_output(Plan)

planner = planner_prompt | planner_llm

replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
对于给定的目标，提出一个简单的逐步计划。这个计划应该包含独立的任务，如果正确执行将得出正确的答案。不要添加任何多余的步骤。最后一步的结果应该是最终答案。确保每一步都有所有必要的信息-
            
你的目标是:
{input}

你的原计划是:
{plan}

你目前已完成的步骤是:
{past_steps}

相应地更新你的计划。如果不需要更多的步骤并且可以返回给用户，那么就这样回应。如果需要，填写计划。只添加仍然需要完成的步骤相应地更新你的计划。如果不需要更多步骤并且可以返给用户，那么就这样回应。如果需要，填写计划。只添加仍然需要完成的步骤。不要返回已完成的步骤作为计划的一部分
            """
        ),
        ("placeholder", "{messages}"),
    ]
)

replanner_llm = ChatTongyi(max_retries=5).with_structured_output(Action)

replanner = replanner_prompt | replanner_llm


async def main():
    async def plan_step(state: PlanExecute):
        plan = await planner.ainvoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}

    async def execute_step(state: PlanExecute):
        plan = state["plan"]
        plant_content = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""
对于以下计划：
{plant_content}\n\n你的任务是执行{1}步，{task}。
"""
        agent_response = await agent_executor.ainvoke({"messages": [("user", task_formatted)]})
        return {
            "past_steps": state["past_steps"] + [(task, agent_response["messages"][-1].content)],
        }

    async def replan_step(state: PlanExecute):
        print(state)
        output = await replanner.ainvoke(state)

        if "action" in output and isinstance(output.action, Response):
            return {"response": output.action.response}
        return {"plan": output.action.steps}

    async def should_end(state: PlanExecute) -> str:
        if "response" in state and state["response"]:
            return END
        return "agent"

    workflow = StateGraph(PlanExecute)

    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replanner", replan_step)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replanner")

    workflow.add_conditional_edges("replanner", should_end)

    app = workflow.compile()

    graph_png = app.get_graph().draw_mermaid_png()
    with open("agent_workflow.png", "wb") as f:
        f.write(graph_png)

    config = {"recursion_limit": 50}
    inputs = {"input": "2024年巴黎奥运会100米自由泳决赛冠军的家乡是哪里?请用中文答"}

    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != END:
                print(f"{k}: {v}")


asyncio.run(main())

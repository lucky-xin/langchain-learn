import base64
import os
import threading
import uuid
from typing import List, Annotated, Any

import requests
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase, TextRequestsWrapper
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import BaseTool, BaseToolkit
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import END, add_messages, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from pydantic import BaseModel, Field
from requests_auth import OAuth2ResourceOwnerPasswordCredentials
from sqlalchemy import create_engine, URL, util
from typing_extensions import TypedDict

from examples.factory.ai_factory import create_ai


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# 定义一个数据模型，用于存储估值模板的信息
class PromptInstructions(BaseModel):
    """估值服务请求模板"""
    brand_name: str = Field(description="品牌名称")
    brand_id: str = Field(description="品牌id")
    model_name: str = Field(description="车型名称")
    model_id: str = Field(description="车型id")
    trim_name: str = Field(description="车型号名称")
    trim_id: str = Field(description="车型号id")
    city_id: str = Field(description="城市id")
    color_id: str = Field(default="Col09", description="颜色id")
    mileage: float = Field(description="行驶里程，单位为万公里，必须大于0，如：2.4，表示2.4万公里")
    reg_time: str = Field(description="车上牌时间，格式为yyyyMMdd,如：20210401")


params = {
    "brand_table_name": "ref_old_brand",
    "model_table_name": "ref_old_model",
    "trim_table_name": "ref_old_basictrim",
    "city_table_name": "ref_old_city",
    "table_names": "ref_old_brand,ref_old_model,ref_old_basictrim,ref_old_city"
}


def create_oauth2() -> OAuth2ResourceOwnerPasswordCredentials:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": os.getenv("OAUTH2_BASIC_AUTH_HEADER"),
        }
    )
    return OAuth2ResourceOwnerPasswordCredentials(
        token_url=os.getenv("OAUTH2_ENDPOINT"),
        username=os.getenv("OAUTH2_USERNAME"),
        password=os.getenv("OAUTH2_PASSWORD"),
        header_name="Authorization",
        header_value="Oauth2 {token}",
        scope="read",
        session=session
    )


def create_prompt(args: dict[str, str] = None) -> BasePromptTemplate:
    msg = """
你是一名专业的SQL语句编写助手，你需通过对用户输入的问题进行深度理解，并按照要求生成SQL。
只能生成查询语句，不能生成任何delete，insert，update等编辑语句。

你需获取的估值参数如下（注意别名匹配）：
- 品牌名称 (用户可能说的别名：brandName，brand_name或者brand name)
- 品牌id (用户可能说的别名：brandId，brand_id或者brand id)
- 车型名称 (用户可能说的别名：modelName，model_name或者model name)
- 车型id (用户可能说的别名：modelId，model_id或者model id)
- 车型号名称 (用户可能说的别名：trim_name)
- 车型号id (用户可能说的别名：trimId，trim_id或者trim id)
- 城市id (用户可能说的别名：cityId，city_id或者city id)
- 颜色id (用户可能说的别名：colorId，color_id或者color id)
- 行驶里程 
- 车上牌时间（格式为yyyyMMdd）

开始时，你应该始终查看{dialect}数据库中表{table_names}的DDL信息，以了解可以查询的内容，不要跳过这一步，你要先执行以下语句获取相关表的模式：SHOW CREATE TABLE table_name。

然后你要按照下文生成SQL。

获取车型号信息SQL生成流程：
    1.一旦用户跟你确认了trim_id，你就需要用trim_id的值填充以下变量，并生成如下SQL，不要跳过这一步:
        SELECT 
            {brand_table_name}.id brand_id, {brand_table_name}.cn_name brand_name, {model_table_name}.cn_name model_name,
            {model_table_name}.id model_id, {trim_table_name}.cn_name trim_name, {trim_table_name}.id trim_id 
        FROM {trim_table_name} JOIN {model_table_name} ON {model_table_name}.id = {trim_table_name}.model_id 
                               JOIN {brand_table_name} ON {model_table_name}.brand_id = {brand_table_name}.id
        WHERE {trim_table_name}.id = '{{{{trim_id}}}}';
    2.不满足步骤1的情况下，你要按照此流程生成SQL。你要对输入的文本进行合理的切分，在完成文本切分之后，你再生成如下SQL（把变量key替换成你识别出来的关键字）：
        SELECT 
            {brand_table_name}.id brand_id, {brand_table_name}.cn_name brand_name, {model_table_name}.cn_name model_name,
            {model_table_name}.id model_id, {trim_table_name}.cn_name trim_name, {trim_table_name}.id trim_id 
        FROM {trim_table_name} JOIN {model_table_name} ON {model_table_name}.id = {trim_table_name}.model_id 
                               JOIN {brand_table_name} ON {model_table_name}.brand_id = {brand_table_name}.id
        WHERE {trim_table_name}.cn_name like '{{{{key}}}}' 
            OR {trim_table_name}.en_name like '{{{{key}}}}' 
            OR {trim_table_name}.id like '{{{{key}}}}' 
        LIMIT 10;

获取城市信息SQL生成流程：
    a.如果用户告诉你城市id,你就需要用city_id填充以下city_id变量，并生成以下SQL获取信息:
        SELECT {city_table_name}.id city_id, {city_table_name}.cn_name city_name FROM {city_table_name} WHERE {city_table_name}.id = '{{{{city_id}}}}';
    b.不满足步骤a情况下，你要深度思考，然后进行合理的切分。在完成文本切分之后，你再生成如下SQL（把变量key替换成你识别出来的关键字） ：
        SELECT {city_table_name}.id city_id, {city_table_name}.cn_name city_name FROM {city_table_name} WHERE cn_name like '{{{{key}}}}' OR en_name like '{{{{key}}}}' OR abbr_cn_name like '{{{{key}}}}' LIMIT 10;

你生成SQL并将执行SQL查询，将数据返回给用户进行确认，不要跳过这一步！不能自己瞎编乱造！确认信息格式如下，你需要用上文查询结果填充brand_name，brand_id，model_name，model_id，trim_name，trim_id，city_name，city_id变量：
    1. **品牌**：{{{{brand_name}}}}（{{{{brand_id}}}}）
    2. **车型**：{{{{model_name}}}}（{{{{model_id}}}}）
    3. **车型号**：{{{{trim_name}}}}（{{{{trim_id}}}}）
    4. **城市**：{{{{city_name}}}}（{{{{city_id}}}}）
    5. **颜色**：目前默认都为：黑色（Col09）
    6. **行驶里程（万公里）**：用户告知的行驶里程；
    
你要严格按照步骤执行，不要跳过任何步骤。如果你无法辨别这些信息，请他们澄清！不要试图疯狂猜测！
当你拿到需要的所有信息并和用户确认之后，你再调用PromptInstructions工具！
""".format(
        brand_table_name=args["brand_table_name"],
        model_table_name=args["model_table_name"],
        trim_table_name=args["trim_table_name"],
        city_table_name=args["city_table_name"],
        dialect=args.get("dialect", "MySQL"),
        table_names=args.get("table_names", []),
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", msg),
            ("placeholder", "{messages}"),
        ]
    )


def create_sql_tool() -> BaseTool:
    url = URL(
        drivername="mysql+pymysql",
        host=os.getenv("MYSQL_HOST"),
        port=3306,
        database=os.getenv("MYSQL_DB"),
        username=os.getenv("MYSQL_USR"),
        password=base64.b64decode(os.getenv("MYSQL_PWD")).decode("utf-8"),
        query=util.immutabledict({
            "charset": "utf8mb4",
        })
    )
    engine = create_engine(
        url=url,
        pool_recycle=3600,
        echo=True
    )
    db = SQLDatabase(
        engine=engine,
        include_tables=params.get("table_names", "").split(","),
        sample_rows_in_table_info=2  # 在提示词中展示的示例数据行数
    )
    return QuerySQLDatabaseTool(db=db)


# TODO 定义一个函数，用于获取生成提示模板所荒的消息，只获取工具调用之后的消息
def create_tool_call_messages(prompt_system: str, state: State):
    reqs = {}
    other_msgs = []
    messages = state["messages"]
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            reqs = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif reqs is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=reqs))] + other_msgs


def check_message_has_tool(state: State):
    messages = state["messages"]
    if not messages or not messages[-1]:
        raise ValueError("No message found in input")
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        raise ValueError("No tool call found in input")
    return last_message


def create_api_spec():
    api_doc = """
openapi: 3.0.0
info:
  title: 二手车估值接口
  description: 二手车估值（新）接口文档
  version: 1.0.0
  contact:
    name: chaoxin.lu
    email: chaoxin.lu@pistonint.com

servers:
  - url: https://openapi.pistonint.com
    description: 生产环境

paths:
  /evaluate:
    post:
      tags:
        - 估值服务
      summary: 二手车估值（新）
      description: 获取二手车估值信息
      operationId: usedCarValuation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ValuationRequest'
            example:
              datas:
                - trimId: "tri26673"
                  mileage: 2.26
                  cityId: "cit00810"
                  regTime: "20220101"
                  colorId: "Col01"
      responses:
        '200':
          description: 成功响应
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ValuationResponse'
              example:
                code: 0
                msg: "success"
                data:
                  - msrp: 231900
                    checker:
                      checkLevel: "normal"
                      checkMsg: "配置缺失正常|品牌检查正常。|Model检查正常。|OEM检查正常。||正常|增量线性模型预测正常。|价差调用成功"
                      status: 1
                    nprice: 0.0
                    sell:
                      valueA: 153600
                      valueB: 142800
                      valueC: 132800
                      valuePctA: 0.66
                      valuePctB: 0.62
                      valuePctC: 0.57
                    buy:
                      valueA: 153600
                      valueB: 142800
                      valueC: 132800
                      valuePctA: 0.66
                      valuePctB: 0.62
                      valuePctC: 0.57
                reqId: "req_123456"
                took: 42
      security:
        - OAuth2: []

components:
  schemas:
    ValuationRequest:
      type: object
      properties:
        datas:
          type: array
          description: 估值数据
          maxItems: 500
          items:
            type: object
            required:
              - trimId
              - mileage
              - cityId
              - regTime
            properties:
              trimId:
                type: string
                description: 型号ID
              mileage:
                type: number
                format: double
                description: 里程（单位：万公里）
              cityId:
                type: string
                description: 城市ID
              regTime:
                type: string
                pattern: '^\d{8}$'
                description: 上牌时间(yyyyMMdd)
              colorId:
                type: string
                description: 颜色ID
                nullable: true

    ValuationResponse:
      type: object
      properties:
        code:
          type: integer
          format: int32
        msg:
          type: string
        data:
          type: array
          items:
            $ref: '#/components/schemas/ValuationData'
        reqId:
          type: string
        took:
          type: integer
          format: int64

    ValuationData:
      type: object
      properties:
        msrp:
          type: number
          format: double
        checker:
          $ref: '#/components/schemas/CheckResult'
        nprice:
          type: number
          format: double
        sell:
          $ref: '#/components/schemas/PriceInfo'
        buy:
          $ref: '#/components/schemas/PriceInfo'

    CheckResult:
      type: object
      properties:
        checkLevel:
          type: string
          enum: [CRITICAL, ERROR, WARNING, NORMAL, NONE]
        checkMsg:
          type: string
        status:
          type: integer
          format: int32

    PriceInfo:
      type: object
      properties:
        valueA:
          type: number
          format: double
        valueB:
          type: number
          format: double
        valueC:
          type: number
          format: double
        valuePctA:
          type: number
          format: double
        valuePctB:
          type: number
          format: double
        valuePctC:
          type: number
          format: double

  securitySchemes:
    OAuth2:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: OAuth2认证
"""
    return api_doc


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        fallbacks=[RunnableLambda(handle_tool_error)],
        exception_key="error"
    )


def create_evaluate_tool() -> BaseToolkit:
    ALLOW_DANGEROUS_REQUEST = True
    return RequestsToolkit(
        requests_wrapper=TextRequestsWrapper(
            auth=create_oauth2(),
            response_content_type="json",
            headers={}
        ),
        allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
    )


def to_tool_message(sur: BaseMessage, tool_call_id: str, prefix: str = ""):
    return ToolMessage(
        tool_call_id=tool_call_id,
        content=prefix + sur.content,
    )


def get_ai_message_tool_name(msg: BaseMessage) -> str:
    if not isinstance(msg, AIMessage) or not msg.tool_calls:
        return ""
    return msg.tool_calls[0].get("name", "")


# 定义一个函数，用于获取当前状态
def get_state(state: State) -> str:
    messages = state.get("messages", [])
    size = len(messages)
    if size > 1:
        last_message = messages[-2]
        if isinstance(last_message, ToolMessage) and last_message.name == "PromptInstructions":
            return "evaluator"
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        if get_ai_message_tool_name(last_message) == "sql_db_query":
            return "sql_tool"
        return END
    elif not isinstance(last_message, HumanMessage):
        return END
    return "info"


def create_evaluate_agent() -> CompiledGraph:
    system_message = """
You have access to an API to help answer user queries.
Here is documentation on the API:
{api_spec}
""".format(api_spec=create_api_spec())
    return create_react_agent(llm, create_evaluate_tool().get_tools(), prompt=system_message)


def get_user_info_chain(state: State):
    resp = user_info_agent.invoke(state)
    return resp


def sql_tool_chain(state: State):
    last_message = check_message_has_tool(state)
    tool_call = last_message.tool_calls[0]
    res = sql_tool.invoke(
        input={
            "type": "tool_call",
            "id": tool_call.get("id"),
            "args": tool_call.get("args", {})
        }
    )
    return {
        "messages": [
            to_tool_message(sur=res, tool_call_id=tool_call["id"], prefix="调用数据库工具获取信息如下：\n")
        ]
    }


def evaluate_chain(state: State):
    # 将消息处理链定义为 get_evaluate_messages 函数和 LLM 实例
    prompt_system = """
基于以下信息，完成估值接口调用，获取车型估值信息；

{reqs}
"""
    messages = create_tool_call_messages(prompt_system, state)
    return evaluate_agent.invoke({"messages": messages})


sql_tool = create_sql_tool()
llm = create_ai()

tools = [PromptInstructions, sql_tool]
# st.session_state.user_info_agent = create_prompt(params) | st.session_state.llm.bind_tools(tools)
user_info_agent = create_react_agent(llm, tools, prompt=create_prompt(params))
evaluate_agent = create_evaluate_agent()

# 初始化 MemorySaver 共例
workflow = StateGraph(State)
workflow.add_node("info", get_user_info_chain)
workflow.add_node("sql_tool", sql_tool_chain)
workflow.add_node("evaluator", evaluate_chain)

workflow.add_conditional_edges(
    source="info",
    path=get_state,
    path_map=[
        "sql_tool",
        "info",
        "evaluator",
        END]
)

workflow.add_edge(START, "info")
workflow.add_edge("sql_tool", "info")
workflow.add_edge("evaluator", END)

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config = {
    "recursion_limit": 50,
    "configurable": {
        "run_id": str(uuid.uuid4()),
        "thread_id": str(threading.current_thread().ident)
    }
}

while True:
    user_input = input("请输入...")
    for chunks in graph.stream(
            input={"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values"
    ):
        messages = chunks.get("messages", [])
        for msg in messages:
            if isinstance(msg, AIMessage):
                msg.pretty_print()

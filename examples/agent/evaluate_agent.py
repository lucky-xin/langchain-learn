import os
import threading
import uuid
from typing import List, Annotated

import requests
import streamlit as st
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase, TextRequestsWrapper
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, BaseMessage, AIMessageChunk, SystemMessage
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.tools import BaseTool, BaseToolkit
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import END, add_messages, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from requests_auth import OAuth2ResourceOwnerPasswordCredentials
from sqlalchemy import create_engine, URL, util
from typing_extensions import TypedDict

from examples.agent.streamlit_callback_utils import get_streamlit_cb
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
}


def create_oauth2() -> OAuth2ResourceOwnerPasswordCredentials:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": os.getenv("OAUTH2_AUTHORIZATION_HEADER"),
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


@st.cache_resource(ttl="1d")
def create_prompt(args: dict[str, str] = None) -> BasePromptTemplate:
    msg = """
你是一名专业的二手车估值助手，需通过和数据库交互精准获取车辆参数，严格遵循以下流程：
开始时，你应该始终查看{dialect}数据库中的表和字段信息，以了解可以查询的内容，不要跳过这一步。

然后，你应该执行以下语句获取相关表的模式。
SHOW CREATE TABLE `{brand_table_name}`;
SHOW CREATE TABLE `{model_table_name}`;
SHOW CREATE TABLE `{trim_table_name}`;

需获取以下参数（注意别名匹配）：
- 车型号id (别名：trimId，trim_id或者trim id)
- 城市id (别名：cityId，city_id或者city id)
- 颜色id(别名：colorId，color_id或者color id)
- 行驶里程 
- 车上牌时间（格式为yyyyMMdd）
    
你的后续工作如下：

获取车型号信息流程：
    1.一旦你能确认了trim_id，你就需要用trim_id的值填充以下trim_id变量，并执行以下SQL获取信息，不要跳过这一步:
        SELECT 
            {brand_table_name}.id brand_id, {brand_table_name}.cn_name brand_name, {model_table_name}.cn_name model_name,
            {model_table_name}.id model_id, {trim_table_name}.cn_name trim_name, {trim_table_name}.id trim_id 
        FROM {trim_table_name} JOIN {model_table_name} ON {model_table_name}.id = {trim_table_name}.model_id 
                               JOIN {brand_table_name} ON {model_table_name}.brand_id = {brand_table_name}.id
        WHERE {trim_table_name}.id = {{{{trim_id}}}}
    2.一旦步骤1中查询到数据你就要进入信息确认流程，让用户进行确认。用户确认之后，你就不要再重复进入获取车型号信息流程。
    3.不满足步骤1和步骤2情况下，你要深度思考，对输入的文本进行识别，并进行合理的切分。再对表{trim_table_name}进行模糊匹配，获取前10条记录。生成查询条件例子如下: 
        SELECT 
            {brand_table_name}.id brand_id, {brand_table_name}.cn_name brand_name, {model_table_name}.cn_name model_name,
            {model_table_name}.id model_id, {trim_table_name}.cn_name trim_name, {trim_table_name}.id trim_id 
        FROM {trim_table_name} JOIN {model_table_name} ON {model_table_name}.id = {trim_table_name}.model_id 
                               JOIN {brand_table_name} ON {model_table_name}.brand_id = {brand_table_name}.id
        WHERE {trim_table_name}.cn_name like '{{{{key}}}}' OR {trim_table_name}.en_name like '{{{{key}}}}' OR {trim_table_name}.id like '{{{{key}}}}' 
        LIMIT 10;
        
        特别注意： 要把变量key替换成你识别出来的关键字；如果查询不到数据，一定不要自己瞎造数据给用户展示；如果通过工具获取成功获取数据，以列表展示给用户，让用户选择并确认具体的车型号id；
    4.如果步骤1和步骤3都查询不到数据，你就提示用户输入车型号相关信息，直到你能在数据库中获取正确的车型号为止；


获取城市id流程：
    1.如果用户告诉你城市id,你就需要用city_id填充以下city_id变量，并执行以下SQL获取信息:
        SELECT 
            {city_table_name}.id city_id, {city_table_name}.cn_name city_name
        FROM {city_table_name} 
        WHERE {city_table_name}.id = '{{{{city_id}}}}'
    2.如果步骤1查询不到数据，你要深度思考，对输入的文本进行识别，然后进行合理的切分。再对表{city_table_name}进行模糊匹配，获取前10条记录。生成查询条件例子如下：
        SELECT 
            {city_table_name}.id city_id, {city_table_name}.cn_name city_name
        FROM {city_table_name} 
        WHERE cn_name like '{{{{key}}}}' OR en_name like '{{{{key}}}}' OR abbr_cn_name like '{{{{key}}}}'
        LIMIT 10;
        
        特别注意： 要把变量key替换成你识别出来的关键字；如果查询不到数据，一定不要自己瞎造数据给用户展示；如果通过工具获取成功获取数据，以列表展示给用户，让用户选择并确认具体的城市id；

信息确认流程：
确认信息格式如下，对于以下变量brand_name、brand_id、model_name、model_id、trim_name、trim_id、city_name、city_id你需要用上文查询结果填充以下内容
    1. **品牌**：{{{{brand_name}}}}（{{{{brand_id}}}}）
    2. **车型**：{{{{model_name}}}}（{{{{model_id}}}}）
    3. **车型号**：{{{{trim_name}}}}（{{{{trim_id}}}}）
    4. **城市**：{{{{city_name}}}}（{{{{city_id}}}}）
    5. **颜色**：目前默认都为：黑色（Col09）
    6. **行驶里程（万公里）**：用户告知的行驶里程；
    不需要确认的信息可以不用显示出来。

你要严格按照步骤执行，不要跳过任何步骤。如果你无法辨别这些信息，请他们澄清！不要试图疯狂猜测！
当你能够辨别所有信息，要跟用户进行确认，不要跳过这一步。
最后，用户确认数据没有问题之后，并调用相关工具。
""".format(
        brand_table_name=args["brand_table_name"],
        model_table_name=args["model_table_name"],
        trim_table_name=args["trim_table_name"],
        city_table_name=args["city_table_name"],
        dialect=args.get("dialect", "MySQL"),
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
        password="",
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
        include_tables=list(params.values()),
        sample_rows_in_table_info=2  # 在提示词中展示的示例数据行数
    )
    return QuerySQLDatabaseTool(db=db)


# 定义一个函数，用于获取生成提示模板所荒的消息，只获取工具调用之后的消息
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
    return messages[-1], [SystemMessage(content=prompt_system.format(reqs=reqs))] + other_msgs


def check_message_has_tool(state: State):
    messages = state["messages"]
    if not messages or not messages[-1]:
        raise ValueError("No message found in input")
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        raise ValueError("No tool call found in input")
    return last_message


@st.cache_resource(ttl="1d")
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


def create_evaluate_agent() -> CompiledGraph:
    system_message = """
You have access to an API to help answer user queries.
Here is documentation on the API:
{api_spec}
""".format(api_spec=create_api_spec())
    return create_react_agent(st.session_state.llm, create_evaluate_tool().get_tools(), prompt=system_message)


def to_tool_message(sur: BaseMessage, prefix: str, tool_call_id: str):
    return ToolMessage(
        tool_call_id=tool_call_id,
        content=prefix + sur.content,
    )


# 定义一个函数，用于获取当前状态
def get_state(state: State) -> str:
    messages = state.get("messages", [])
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        if last_message.tool_calls[0].get("name") == "sql_db_query":
            return "sql_tool"
        return "evaluator"
    elif not isinstance(last_message, HumanMessage):
        return END
    return "info"


def get_user_info_chain(state: State):
    print("-------------------------------")
    for e in state["messages"]:
        e.pretty_print()

    resp = st.session_state.user_info_chain.invoke(state)
    return {"messages": [resp]}


def sql_tool_chain(state: State):
    last_message = check_message_has_tool(state)
    tool_call = last_message.tool_calls[0]
    res = st.session_state.sql_tool.invoke(
        input={"type": "tool_call",
               "id": tool_call.get("id"),
               "args": tool_call.get("args", {})
               }
    )
    return {
        "messages": [
            to_tool_message(sur=res, prefix="调用数据库工具获取信息如下：\n", tool_call_id=tool_call["id"])
        ]
    }


def evaluate_chain(state: State):
    # 将消息处理链定义为 get_evaluate_messages 函数和 LLM 实例
    prompt_system = """
基于以下信息，完成估值接口调用，获取车型估值信息；

{reqs}
"""
    last_message, messages = create_tool_call_messages(prompt_system, state)
    resp = st.session_state.evaluate_agent.invoke({"messages": messages})
    if resp.get("messages") and resp.get("messages")[-1]:
        last_msg = resp.get("messages")[-1]
        last_msg.pretty_print()
        return {
            "messages": [
                to_tool_message(
                    sur=last_msg,
                    prefix="调用估值接口获取信息如下：\n",
                    tool_call_id=last_message.tool_calls[0]["id"]
                )
            ]
        }
    return {"messages": []}


# Create graph builder
if not st.session_state.get("graph"):
    print("Creating graph...")
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

    # checkpointer = create_checkpointer()
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    st.session_state.graph = graph
    st.session_state.run_id = str(uuid.uuid4())

st.image(
    image=st.session_state.graph.get_graph().draw_mermaid_png(),
    caption="二手车估值助手流程",
    use_container_width=False
)


def init():
    print("init...")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if "sql_tool" not in st.session_state:
        st.session_state.sql_tool = create_sql_tool()
    if "llm" not in st.session_state:
        st.session_state.llm = create_ai()
    if "llm_with_tool" not in st.session_state:
        st.session_state.llm_with_tool = st.session_state.llm.bind_tools(
            [PromptInstructions, st.session_state.sql_tool]
        )
    if "user_info_chain" not in st.session_state:
        st.session_state.user_info_chain = create_prompt(params) | st.session_state.llm_with_tool
    if "evaluate_agent" not in st.session_state:
        st.session_state.evaluate_agent = create_evaluate_agent()


def invoke(user_input: str):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    collected_messages = ""
    with st.chat_message("assistant"):
        st_cb = get_streamlit_cb(st.container())
        config = {
            "recursion_limit": 50,
            "configurable": {
                "run_id": st.session_state.run_id,
                "thread_id": str(threading.current_thread().ident)
            },
            "callbacks": [st_cb]
        }
        output_placeholder = st.empty()
        for chunks in st.session_state.graph.stream(
                input={"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
        ):
            for chunk in chunks:
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    collected_messages += chunk.content
                    output_placeholder.markdown(collected_messages + "▌")
            output_placeholder.markdown(collected_messages)
        st.session_state.messages.append({"role": "assistant", "content": collected_messages})


init()
if q := st.chat_input("请输入需求信息"):
    invoke(q)

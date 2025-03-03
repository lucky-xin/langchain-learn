import base64
import os
import threading
import uuid
from typing import List, Annotated, Any

import requests
import streamlit as st
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.tools import QuerySQLDatabaseTool, InfoSQLDatabaseTool
from langchain_community.utilities import SQLDatabase, TextRequestsWrapper
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, AIMessageChunk, SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import BaseTool, BaseToolkit
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import END, add_messages, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from pydantic import BaseModel, Field
from requests_auth import OAuth2ResourceOwnerPasswordCredentials
from sqlalchemy import create_engine, URL, util
from typing_extensions import TypedDict

from examples.agent.streamlit_callback_utils import get_streamlit_cb
from examples.agent.tools.car_evaluator import CarEvaluateTool
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
    trim_id: str = Field(description="车型号id", pattern="^tri(\d+)$")
    city_id: str = Field(description="城市id", pattern="^cit(\d+)$")
    color_id: str = Field(default="Col09", description="颜色id", pattern="^Col(\d+)$")
    mileage: float = Field(description="行驶里程，单位为万公里，必须大于0，如：2.4，表示2.4万公里,最大值50")
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
你要严格按照步骤执行，不要跳过任何步骤。如果你无法辨别这些信息，请他们澄清！不要试图疯狂猜测！不要自己瞎造数据！
当你拿到需要的所有信息并和用户确认之后，你再调用PromptInstructions工具！

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

开始时，你要始终查看{dialect}数据库中表{table_names}的DDL信息，以了解可以查询的内容，不要跳过这一步。

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
            OR {trim_table_name}.abbr_cn_name like '{{{{key}}}}' 
            OR {trim_table_name}.abbr_en_name like '{{{{key}}}}' 
            OR {trim_table_name}.id like '{{{{key}}}}' 
        LIMIT 10;
        
    你要以markdown格式把车型号列表展示给用户，让用户进行选择。你需要用上文查询结果填充以下brand_name，brand_id，model_name，model_id，trim_name，trim_id变量，例子如下：
    
        | 品牌名称    |  品牌ID     | 车型名称     |   车型ID  |  车型号名称 | 车型号ID   |
        |------------|------------|-------------|----------|------------|----------|
        | brand_name | brand_id   | model_name  | model_id | trim_name  | trim_id  |
    

获取城市信息SQL生成流程：
    a.如果用户告诉你城市id,你就需要用city_id填充以下city_id变量，并生成以下SQL获取信息:
        SELECT {city_table_name}.id city_id, {city_table_name}.cn_name city_name FROM {city_table_name} WHERE {city_table_name}.id = '{{{{city_id}}}}';
    b.不满足步骤a情况下，你要深度思考，然后进行合理的切分。在完成文本切分之后，你再生成如下SQL（把变量key替换成你识别出来的关键字） ：
        SELECT {city_table_name}.id city_id, {city_table_name}.cn_name city_name FROM {city_table_name} WHERE cn_name like '{{{{key}}}}' OR en_name like '{{{{key}}}}' OR abbr_cn_name like '{{{{key}}}}' LIMIT 10;

估值前的数据确认要求：你生成SQL并将执行SQL查询，将数据返回给用户进行确认，不要跳过这一步！不能自己瞎编乱造数据给用户！
你需要用上文查询结果填充以下brand_name，brand_id，model_name，model_id，trim_name，trim_id，city_name，city_id变量，并按照以下例子进行展示：
    1. **品牌**：{{{{brand_name}}}}（{{{{brand_id}}}}）
    2. **车型**：{{{{model_name}}}}（{{{{model_id}}}}）
    3. **车型号**：{{{{trim_name}}}}（{{{{trim_id}}}}）
    4. **城市**：{{{{city_name}}}}（{{{{city_id}}}}）
    5. **颜色**：目前默认都为：黑色（Col09）
    6. **行驶里程（万公里）**：用户告知的行驶里程；
    
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


def create_sql_tools() -> [BaseTool]:
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
    return [
        QuerySQLDatabaseTool(db=db),
        InfoSQLDatabaseTool(db=db)
    ]


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
    return [SystemMessage(content=prompt_system.format(reqs=reqs))] + other_msgs


def check_message_has_tool(state: State):
    messages = state["messages"]
    if not messages or not messages[-1]:
        raise ValueError("No message found in input")
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        raise ValueError("No tool call found in input")
    return last_message


def create_vim_match_api_spec():
    api_doc = """
openapi: 3.0.3
info:
  title: VIN 匹配 API
  description: 根据VIN码查询车辆信息的接口
  version: 1.0.0
  contact:
    name: chaoxin.lu

servers:
  - url: https://openapi.pistonint.com
    description: 生产环境

paths:
  /vin/match:
    get:
      tags:
        - VIN
      summary: VIN 匹配
      description: 根据VIN码查询车辆信息
      operationId: vinMatch
      parameters:
        - name: id
          in: query
          description: vin码
          required: true
          schema:
            type: string
          example: LSGUD84X5GE013971
        - name: Authorization
          in: header
          description: OAuth2 认证令牌
          required: true
          schema:
            type: string
          example: OAuth2 xxx
      responses:
        '200':
          description: 成功响应
          content:
            application/json:
              schema:
                type: object
                required:
                  - code
                  - msg
                  - data
                  - reqId
                  - took
                properties:
                  code:
                    type: integer
                    format: int64
                    example: 1
                  msg:
                    type: string
                    example: Success.
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/VehicleInfo'
                  reqId:
                    type: string
                    example: 77f639819d86a99032107c2ba7b98f7d
                  took:
                    type: integer
                    format: int64
                    example: 483

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
    VehicleInfo:
      type: object
      required:
        - id
        - cnName
        - enName
        - abbrEnName
        - abbrCnName
        - valid
        - year
        - msrp
        - onSaleFlag
        - isClassic
        - stopTime
        - startTime
        - bodyTypeId
        - modelId
        - generation
        - oemId
        - brandId
        - oemName
        - brandName
        - modelName
        - bodyType
        - segmentType
        - delFlag
        - name
        - abbrName
      properties:
        id:
          type: string
          example: tri04003
        cnName:
          type: string
          example: 上汽通用 别克GL8 2014款 2.4L 经典型
        enName:
          type: string
          example: SAIC-GM Buick GL8 2014 2.4L Classic
        abbrEnName:
          type: string
          example: 2014 2.4L Classic
        abbrCnName:
          type: string
          example: 2014款 2.4L 经典型
        valid:
          type: boolean
          example: true
        year:
          type: integer
          example: 2014
        msrp:
          type: integer
          example: 209000
        onSaleFlag:
          type: boolean
          example: false
        isClassic:
          type: boolean
          example: false
        stopTime:
          type: string
          example: 20151001
        startTime:
          type: string
          example: 20131201
        bodyTypeId:
          type: string
          example: bod00050
        modelId:
          type: string
          example: mod02470
        generation:
          type: string
          example: Model_Year
        oemId:
          type: string
          example: oem01320
        brandId:
          type: string
          example: bra00220
        oemName:
          type: string
          example: 上汽通用
        brandName:
          type: string
          example: 别克
        modelName:
          type: string
          example: 别克GL8
        bodyType:
          type: string
          example: MPV
        segmentType:
          type: string
          example: AutoHome
        delFlag:
          type: integer
          example: 0
        name:
          type: string
          example: 上汽通用 别克GL8 2014款 2.4L 经典型
        abbrName:
          type: string
          example: 2014款 2.4L 经典型
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


def create_requests_toolkit() -> BaseToolkit:
    ALLOW_DANGEROUS_REQUEST = True
    return RequestsToolkit(
        requests_wrapper=TextRequestsWrapper(
            auth=create_oauth2(),
            response_content_type="json",
            headers={}
        ),
        allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
    )


def to_tool_message(sur: BaseMessage, prefix: str, tool_call_id: str):
    return ToolMessage(
        tool_call_id=tool_call_id,
        content=prefix + sur.content,
    )


def get_ai_message_tool_name(msg: BaseMessage) -> str:
    return msg.tool_calls[0].get("name", "") if not isinstance(msg, AIMessage) or not msg.tool_calls else ""


def get_user_info_chain(state: State):
    resp = st.session_state.user_info_agent.invoke(state)
    return resp


def init():
    print("init...")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": """
        您好！我是一名专业估值助手，为了获取车辆估值数据，你需要请输入以下参数，如果您不确定是哪一个车型号，我们就一切来确定。
        - 车型号名称
        - 城市
        - 颜色
        - 行驶里程
        - 车上牌时间（年月日）
        """}
        ]
    if "llm" not in st.session_state:
        st.session_state.llm = create_ai()
    if "user_info_agent" not in st.session_state:
        car_evaluate_tool = CarEvaluateTool(auth=create_oauth2())
        tools = [PromptInstructions, car_evaluate_tool] + create_sql_tools()
        st.session_state.user_info_agent = create_react_agent(st.session_state.llm, tools, prompt=create_prompt(params))
    # Create graph builder
    if not st.session_state.get("graph"):
        print("Creating graph...")
        # 初始化 MemorySaver 共例
        workflow = StateGraph(State)
        workflow.add_node("info", get_user_info_chain)
        workflow.add_edge(START, "info")
        workflow.add_edge("info", END)

        # checkpointer = create_checkpointer()
        checkpointer = MemorySaver()
        graph = workflow.compile(checkpointer=checkpointer)
        st.session_state.graph = graph
        st.session_state.run_id = str(uuid.uuid4())
        st.session_state.image = st.session_state.graph.get_graph().draw_mermaid_png()

    st.image(
        image=st.session_state.image,
        caption="二手车估值助手流程",
        use_container_width=False
    )
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


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

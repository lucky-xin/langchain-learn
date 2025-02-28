import os
import threading
import uuid
from typing import List, Annotated

from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase, TextRequestsWrapper
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, BaseMessage, AIMessageChunk
from langchain_core.messages import SystemMessage
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.tools import BaseTool, BaseToolkit
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import END, add_messages, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
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
    mileage: float = Field(description="行驶里程，单位为万公里，如：2.4，表示2.4万公里")
    reg_time: str = Field(description="车上牌时间，格式为yyyyMMdd,如：20210401")


def create_prompt(args: dict[str, str] = None) -> BasePromptTemplate:
    msg = """
你是一个二手车估值服务助手，你的工作是从用户那里获取估值请求API所需的参数：
    - 车型号id (可能的其他叫法：trimId，trim_id或者trim id)
    - 城市id (可能的其他叫法：cityId，city_id或者city id)
    - 颜色id(可能的其他叫法：colorId，color_id或者color id)
    - 行驶里程
    - 车上牌时间                      
你要尽最大努力获取足够多的信息，以完成下一步工作，如果你无法辨别这些信息，请他们澄清！不要试图疯狂猜测！

确认车型号步骤如下：
    1.一旦你能确认了车型号id(trim_id)，你就需要用trim_id填充以下trim_id变量，并执行以下SQL获取信息:
        SELECT 
            {brand_table_name}.id brand_id, {brand_table_name}.cn_name brand_name, {model_table_name}.cn_name model_name,
            {model_table_name}.id model_id, {trim_table_name}.cn_name trim_name, {trim_table_name}.id trim_id 
        FROM {trim_table_name} JOIN {model_table_name} ON {model_table_name}.id = {trim_table_name}.model_id 
                               JOIN {brand_table_name} ON {model_table_name}.brand_id = {brand_table_name}.id
        WHERE {trim_table_name}.id = {{{{trim_id}}}}
    2.一旦步骤1中查询到数据你就要进入信息确认流程，让用户进行确认。用户确认之后，你就不要再重复获取车型号流程；
    3.如果用户没有告诉你车型号id，你要对输入的文本进行识别,然后进行合理的切分，再去表{trim_table_name}中对cn_name和en_name和id进行模糊匹配，获取前10条记录，生成查询条件例子如下，要把XX替换成你识别出来的关键字: 
        SELECT 
            {trim_table_name}.cn_name trim_name, {trim_table_name}.id trim_id 
        FROM {trim_table_name}
        WHERE cn_name like '%XX%' OR en_name like '%XX%' OR id like '%XX%' 
        LIMIT 10
        查询结果集为多个时你就要以列表展示给用户，让用户选择并确认具体的车型号id，如果查询不到数据就不要给用户展示结果。
    4.如果步骤1和步骤3都查询不到数据，你就提示用户输入车型号相关信息，直到你能在数据库中获取正确的车型号为止；
    
确认城市id步骤如下：
    1.如果用户告诉你城市id,你就需要用city_id填充以下city_id变量，并执行以下SQL获取信息:
        SELECT 
            {city_table_name}.id city_id, {city_table_name}.cn_name city_name
        FROM {city_table_name} 
        WHERE {city_table_name}.id = {{{{city_id}}}}
    2.如果步骤1查询不到数据，你要对输入的文本进行识别,然后进行合理的切分。然后对表{city_table_name}中cn_name、en_name和abbr_cn_name进行模糊匹配，获取前10条记录，生成查询SQL如下，要把XX替换成你识别出来的关键字:
        SELECT 
            {city_table_name}.id city_id, {city_table_name}.cn_name city_name
        FROM {city_table_name} 
        WHERE cn_name like '%XX%' OR en_name like '%XX%' OR abbr_cn_name like '%XX%'
        LIMIT 10
        查询结果集为多个时你就要以列表展示给用户，让用户选择并确认具体的城市id，如果查询不到数据就不要给用户展示结果。

确认信息格式如下，对于以下变量brand_name、brand_id、model_name、model_id、trim_name、trim_id、city_name、city_id你需要用上文查询结果填充以下内容
    1. **品牌**：{{{{brand_name}}}}（{{{{brand_id}}}}）
    2. **车型**：{{{{model_name}}}}（{{{{model_id}}}}）
    3. **车型号**：{{{{trim_name}}}}（{{{{trim_id}}}}）
    4. **城市**：{{{{city_name}}}}（{{{{city_id}}}}）
    5. **颜色**：目前默认都为：黑色（Col09）
    6. **行驶里程（万公里）**：用户告知的行驶里程；
    不需要确认的信息可以不用显示出来。
SQL表元数据如下：
    品牌表-{brand_table_name}：
        id varchar(32) '品牌id'；
        abbr_en_name varchar(100) '品牌英文全称'；
        en_name varchar(100) '品牌英文全称'；
        cn_name varchar(100) '品牌中文全称'；
        valid varchar(10) '数据是否有效，true为有效'；
    车型表-{model_table_name}：
        id varchar(32) '主键'
        cn_name varchar(100) '英文名称'
        en_name varchar(100) '中文名称'
        brand_id varchar(32) '表ref_old_brand的主键',
        on_sale_flag varchar(10) '是否在售，true表示在售'
        valid varchar(100) '是否有效，true表示数据有效'
    车型号表-{trim_table_name}：
        id varchar(32) '主键';
        cn_name varchar(1000) '车型号中文全称';
        en_name varchar(1000) '车型号英文全称';
        abbr_cn_name varchar(1000) '车型号中文简称';
        abbr_en_name varchar(1000) '车型号英文简称';
        on_sale_flag varchar(10) '是否在售标识，true表示在售';
        model_id varchar(100) '表ref_old_model的主键';
        valid varchar(100) '是否有效，true表示数据有效'
    城市表-{city_table_name}:
        id varchar(32) '主键'；
        valid varchar(100) '是否有效，true表示数据有效'；
        abbr_cn_name varchar(32) '中文简称'；
        cn_name varchar(32) '中文全称'；
        en_name varchar(100) '英文全称'；
当你能够辨别所有信息，并跟用户确认数据没有问题之后，并调用相关工具；
""".format(
        brand_table_name=args["brand_table_name"],
        model_table_name=args["model_table_name"],
        trim_table_name=args["trim_table_name"],
        city_table_name=args["city_table_name"],
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", msg),
            ("placeholder", "{messages}"),
        ]
    )


# 定义一个新的系统提示模板
prompt_system = """
基于以下信息，完成估值接口调用，获取车型估值信息；

{reqs}
"""
params = {
    "brand_table_name": "ref_old_brand",
    "model_table_name": "ref_old_model",
    "trim_table_name": "ref_old_basictrim",
    "city_table_name": "ref_old_city",
}


def create_sql_tool() -> BaseTool:
    url = URL(
        drivername="mysql+pymysql",
        host=os.getenv("MYSQL_HOST"),
        port=3306,
        database=os.getenv("MYSQL_DB"),
        username=os.getenv("MYSQL_USR"),
        password=os.getenv("MYSQL_PWD"),
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


sql_tool = create_sql_tool()

llm = create_ai()
llm_with_tool = llm.bind_tools([PromptInstructions, sql_tool])
user_info_chain = create_prompt(params) | llm_with_tool


# 定义一个函数，用于获取生成提示模板所荒的消息，只获取工具调用之后的消息
def get_evaluate_messages(state: State):
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


def create_api_spec():
    api_doc = """
openapi: 3.0.3
info:
  title: Vehicle Evaluation API
  version: 1.0.0
  description: 车辆评估接口文档

servers:
  - url: https://openapi.pistonint.com
    description: Production server

paths:
  /evaluate:
    post:
      summary: 获取车辆评估结果
      description: 根据车辆信息计算评估值
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - trimId
                - cityId
                - mileage
                - regTime
              properties:
                trimId:
                  type: string
                  description: 车型号ID
                  example: "tri00003"
                cityId:
                  type: string
                  description: 城市ID
                  example: "cit00010"
                colorId:
                  type: string
                  description: 颜色ID
                  default: "Col09"
                  example: "Col12"
                mileage:
                  type: number
                  format: float
                  description: 行驶里程（公里）
                  example: 10.5
                regTime:
                  type: string
                  format: date
                  pattern: '^\d{8}$'
                  description: 上牌日期（yyyyMMdd格式）
                  example: "20210401"
      responses:
        '200':
          description: 评估结果返回成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  code:
                    type: integer
                    example: 200
                  data:
                    type: object
                    properties:
                      valuation:
                        type: number
                        format: float
                        example: 235000.00

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: 使用Bearer Token认证
"""
    return api_doc


def create_evaluate_tool() -> BaseToolkit:
    ALLOW_DANGEROUS_REQUEST = True
    return RequestsToolkit(
        requests_wrapper=TextRequestsWrapper(
            response_content_type="json",
            headers={}
        ),
        allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
    )


# 定义一个函数，用于获取当前状态
def get_state(state: State) -> str:
    messages = state.get("messages", [])
    print("get_state------------------- start")
    for msg in messages:
        print("get_state.pretty_print------------------- start")
        msg.pretty_print()
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        if last_message.tool_calls[0].get("name") == "sql_db_query":
            print("get_state------------------- to add_sql_tool node")
            return "add_sql_tool"
        return "add_evaluate_tool"
    elif not isinstance(last_message, HumanMessage):
        return END
    print("get_state------------------- to info node")
    return "info"


def get_user_info_chain(state: State):
    print("get_user_info_chain-------------------")
    messages = state["messages"]
    print("get_user_info_chain------------------- start")
    for msg in messages:
        print("get_user_info_chain.pretty_print------------------- start")
        msg.pretty_print()
    print("get_user_info_chain.pretty_print------------------- end")
    resp = user_info_chain.invoke({"messages": messages})

    print("get_user_info_chain------------------- resp")
    print(resp.pretty_print())
    return {"messages": [resp]}


def add_sql_tool(state: State):
    print("add_sql_tool-------------------")
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    res = sql_tool.invoke(
        input={
            "type": "tool_call",
            "id": tool_call["id"],
            "args": tool_call["args"]
        }
    )
    return {
        "messages": [
            ToolMessage(
                tool_call_id=tool_call["id"],
                content="调用数据库工具获取信息如下：\n" + res.content,
            )
        ]
    }


def add_evaluate_tool(state: State):
    print("add_evaluate_tool-------------------")
    return {
        "messages": [
            ToolMessage(
                content="调用估值API获取估值信息",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }


def evaluate_chain(state: State):
    # 将消息处理链定义为 get_evaluate_messages 函数和 LLM 实例
    print("evaluate_chain-------------------")
    messages = get_evaluate_messages(state)
    agent_executor = create_evaluate_agent()
    resp = agent_executor.invoke({"messages": messages})
    return {"messages": [resp]}


def create_evaluate_agent() -> CompiledGraph:
    system_message = """
    You have access to an API to help answer user queries.
    Here is documentation on the API:
    {api_spec}
    """.format(api_spec=create_api_spec())
    return create_react_agent(llm, create_evaluate_tool().get_tools(), prompt=system_message)


print("Creating graph...")
# 初始化 MemorySaver 共例
workflow = StateGraph(State)
workflow.add_node("info", get_user_info_chain)
workflow.add_node("add_sql_tool", add_sql_tool)
workflow.add_node("add_evaluate_tool", add_evaluate_tool)
workflow.add_node("evaluator", evaluate_chain)

workflow.add_conditional_edges(source="info",
                               path=get_state,
                               path_map=[
                                   "add_sql_tool",
                                   "info",
                                   "add_evaluate_tool",
                                   END]
                               )

workflow.add_edge(START, "info")
workflow.add_edge("add_sql_tool", "info")
workflow.add_edge("add_evaluate_tool", "evaluator")
workflow.add_edge("evaluator", END)

# checkpointer = create_checkpointer()
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

collected_messages = ""

config = {
    "recursion_limit": 50,
    "configurable": {
        "run_id": str(uuid.uuid4()),
        "thread_id": str(threading.current_thread().ident)
    }
}

while True:
    user_input = input("请输入您的问题")
    for chunks in graph.stream(
            input={"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
    ):
        for chunk in chunks:
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                collected_messages += chunk.content
    if collected_messages:
        print(collected_messages)

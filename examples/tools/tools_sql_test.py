import base64
import os

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from sqlalchemy import create_engine, URL, util

from examples.factory.llm import LLMFactory, LLMType


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

你要严格按照步骤执行，不要跳过任何步骤。如果你无法辨别这些信息，请他们澄清！不要试图疯狂猜测！


你生成SQL并将执行SQL查询，将数据返回给用户进行确认，用户确认没有问题之后，你再更新状态！不要跳过这一步！不能自己瞎编乱造！确认信息格式如下，你需要用上文查询结果填充以下变量：
    1. **品牌**：{{{{brand_name}}}}（{{{{brand_id}}}}）
    2. **车型**：{{{{model_name}}}}（{{{{model_id}}}}）
    3. **车型号**：{{{{trim_name}}}}（{{{{trim_id}}}}）
    4. **城市**：{{{{city_name}}}}（{{{{city_id}}}}）
    5. **颜色**：目前默认都为：黑色（Col09）
    6. **行驶里程（万公里）**：用户告知的行驶里程；

你要严格按照步骤执行，不要跳过任何步骤。如果你无法辨别这些信息，请他们澄清！不要试图疯狂猜测！

当你拿到估值需要的所有信息之后，你仍要跟用户进行确认，不要跳过这一步。
最后，用户确认数据没有问题之后，你再调用相关工具，不要跳过这一步，不要自行调用估值接口！
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
    include_tables=["ref_old_brand", "ref_old_model", "ref_old_basictrim", "ref_old_city"],  # 白名单过滤
    sample_rows_in_table_info=2  # 在提示词中展示的示例数据行数
)
params = {
    "brand_table_name": "ref_old_brand",
    "model_table_name": "ref_old_model",
    "trim_table_name": "ref_old_basictrim",
    "city_table_name": "ref_old_city",
    "table_names": "ref_old_brand,ref_old_model,ref_old_basictrim,ref_old_city"
}
llm_factory = LLMFactory(
    llm_type=LLMType.LLM_TYPE_QWENAI,
)
chat_llm = llm_factory.create_chat_llm()
toolkit = SQLDatabaseToolkit(db=db, llm=chat_llm)

tools = toolkit.get_tools()

prompt_template = create_prompt(params)
prompt_template.pretty_print()

system_message = prompt_template.format(dialect="MySQL", top_k=5)

from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(chat_llm, tools, prompt=system_message)

while True:
    user_input = input("用户：")
    if user_input == "exit":
        break
    for step in agent_executor.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

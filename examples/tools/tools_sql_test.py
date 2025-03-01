import os

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from sqlalchemy import create_engine, URL, util

from examples.factory.ai_factory import create_ai


def create_prompt(args: dict[str, str] = None) -> BasePromptTemplate:
    # 指令模板
    prompt_template = """
你是一个二手车估值服务助手，你的工作是从用户那里获取估值请求API所需的参数：
    - 车型号id (可能的其他叫法：trimId，trim_id或者trim id)
    - 城市id (可能的其他叫法：cityId，city_id或者city id)
    - 颜色id(可能的其他叫法：colorId，color_id或者color id)
    - 行驶里程
    - 车上牌时间                      
你要尽最大努力获取足够多的信息，以完成下一步工作，你要严格按照步骤执行，不要跳过任何步骤。如果你无法辨别这些信息，请他们澄清！不要试图疯狂猜测！

确认车型号步骤如下：
    1.一旦你能确认了车型号id(trim_id)，你就需要用trim_id填充以下trim_id变量，并执行以下SQL获取信息:
        SELECT 
            {brand_table_name}.id brand_id, {brand_table_name}.cn_name brand_name, {model_table_name}.cn_name model_name,
            {model_table_name}.id model_id, {trim_table_name}.cn_name trim_name, {trim_table_name}.id trim_id 
        FROM {trim_table_name} JOIN {model_table_name} ON {model_table_name}.id = {trim_table_name}.model_id 
                               JOIN {brand_table_name} ON {model_table_name}.brand_id = {brand_table_name}.id
        WHERE {trim_table_name}.id = {{{{trim_id}}}}
    2.一旦步骤1中查询到数据你就要进入信息确认流程，让用户进行确认。用户确认之后，你就不要再重复获取车型号流程。
    3.不满足步骤1和步骤2情况下，你要深度思考，对输入的文本进行识别，然后进行合理的切分。再对表{trim_table_name}进行模糊匹配，获取前10条记录。生成查询条件例子如下: 
        SELECT 
            {trim_table_name}.cn_name trim_name, {trim_table_name}.id trim_id 
        FROM {trim_table_name}
        WHERE cn_name like '{{{{key}}}}' OR en_name like '{{{{key}}}}' OR id like '{{{{key}}}}' 
        LIMIT 10;
        特别注意： a.要把变量key替换成你识别出来的关键字，避免'%'号重复；b.如果查询不到数据，一定不要自己瞎造数据给用户展示；c.如果通过工具获取成功获取数据，以列表展示给用户，让用户选择并确认具体的车型号id；
    4.如果步骤1和步骤3都查询不到数据，你就提示用户输入车型号相关信息，直到你能在数据库中获取正确的车型号为止；
    
确认城市id步骤如下：
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
        特别注意： a.要把变量key替换成你识别出来的关键字，避免'%'号重复；b.如果查询不到数据，一定不要自己瞎造数据给用户展示；c.如果通过工具获取成功获取数据，以列表展示给用户，让用户选择并确认具体的城市id；

确认信息格式如下，对于以下变量brand_name、brand_id、model_name、model_id、trim_name、trim_id、city_name、city_id你需要用上文查询结果填充以下内容
    1. **品牌**：{{{{brand_name}}}}（{{{{brand_id}}}}）
    2. **车型**：{{{{model_name}}}}（{{{{model_id}}}}）
    3. **车型号**：{{{{trim_name}}}}（{{{{trim_id}}}}）
    4. **城市**：{{{{city_name}}}}（{{{{city_id}}}}）
    5. **颜色**：目前默认都为：黑色（Col09）
    6. **行驶里程（万公里）**：用户告知的行驶里程；
    不需要确认的信息可以不用显示出来。

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables.
    """.format(
        brand_table_name=args["brand_table_name"],
        model_table_name=args["model_table_name"],
        trim_table_name=args["trim_table_name"],
        city_table_name=args["city_table_name"],
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("placeholder", "{messages}"),
        ]
    )


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
    include_tables=["ref_old_brand", "ref_old_model", "ref_old_basictrim", "ref_old_city"],  # 白名单过滤
    sample_rows_in_table_info=2  # 在提示词中展示的示例数据行数
)
params = {
    "brand_table_name": "ref_old_brand",
    "model_table_name": "ref_old_model",
    "trim_table_name": "ref_old_basictrim",
    "city_table_name": "ref_old_city",
}

toolkit = SQLDatabaseToolkit(db=db, llm=create_ai())

tools = toolkit.get_tools()

prompt_template = create_prompt(params)
prompt_template.pretty_print()

system_message = prompt_template.format(dialect="MySQL", top_k=5)

from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(create_ai(), tools, prompt=system_message)

question = "查询奥迪 2021款 40 TFSI 进享人生版"

for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
):
    step["messages"][-1].pretty_print()

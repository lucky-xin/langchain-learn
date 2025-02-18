import os
import uuid

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.llms import Tongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = Tongyi()

# 定义 ChatPromptTemplate
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    # 历史消息占位符
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}"),
])
session_id = uuid.uuid4().hex
# 创建链式调用
chain = template | llm | StrOutputParser

# PostgreSQL 连接字符串
sql_user = os.getenv("SQL_USER")
sql_pwd = os.getenv("SQL_PWD")
sql_host = os.getenv("SQL_HOST")
sql_port = os.getenv("SQL_PORT")
sql_db = os.getenv("SQL_DB")
connection_string = f"postgresql+psycopg2://${sql_user}:${sql_pwd}@${sql_host}:${sql_port}/${sql_db}??sslmode=disable"
# 创建 SQLChatMessageHistory 实例
chat_history = SQLChatMessageHistory(
    session_id=session_id,
    connection_string=connection_string
)
# 格式化输入
input_data = {"name": "Bob", "user_input": "给我讲一个埃及法老的笑话"}

chat_history.add_messages(template.messages)

# 调用链并获取结果
result = chain.invoke(input_data)
print(result)

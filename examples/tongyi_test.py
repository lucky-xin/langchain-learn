import os
import uuid

import psycopg
from langchain_community.llms import Tongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_postgres import PostgresChatMessageHistory


def create_chat_history(table_name: str):
    # PostgreSQL 连接字符串
    sql_user = os.getenv("SQL_USER")
    sql_pwd = os.getenv("SQL_PWD")
    sql_url = os.getenv("SQL_URL")
    sql_db = os.getenv("SQL_DB")
    connection_string = f"postgresql://{sql_user}:{sql_pwd}@{sql_url}/{sql_db}?sslmode=disable"
    sync_connection = psycopg.connect(connection_string)
    # Create the table schema (only needs to be done once)
    PostgresChatMessageHistory.create_tables(sync_connection, table_name)

    # 创建 SQLChatMessageHistory 实例
    return PostgresChatMessageHistory(
        table_name,
        session_id,
        sync_connection=sync_connection
    )


if __name__ == '__main__':
    llm = Tongyi()

    # 定义 ChatPromptTemplate
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        # 历史消息占位符
        # MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}"),
    ])
    session_id = str(uuid.uuid4())
    # 创建链式调用
    chain = template | llm

    # 格式化输入
    input_data = {"name": "Bob", "user_input": "给我讲一个埃及法老的笑话"}


    # 调用链并获取结果
    result = chain.invoke(input_data)
    print(result)
    table_name = "chat_history"
    chat_history = create_chat_history(table_name)
    chat_history.add_messages(template.messages)

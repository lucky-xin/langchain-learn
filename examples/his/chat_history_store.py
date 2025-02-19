import os
from datetime import datetime
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

import psycopg
from cache3 import Cache
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_postgres import PostgresChatMessageHistory


def get_current_timestamp() -> int:
    # 获取当前时间戳（带时区）
    timezone = ZoneInfo("Asia/Shanghai")  # 例如，上海时区
    current_time = datetime.now(timezone)
    return int(current_time.timestamp())


class PgSQLChatMessageHistory(PostgresChatMessageHistory):
    def __init__(
            self,
            table_name: str,
            session_id: str,
            /,
            *,
            sync_connection: Optional[psycopg.Connection] = None,
            async_connection: Optional[psycopg.AsyncConnection] = None,
    ) -> None:
        super().__init__(
            table_name,
            session_id,
            sync_connection=sync_connection,
            async_connection=async_connection,
        )

    def add_message(self, message: BaseMessage) -> None:
        # message.additional_kwargs["timestamp"] = get_current_timestamp()
        super().add_message(message)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # for message in messages:
        #     message.additional_kwargs["timestamp"] = get_current_timestamp()
        super().add_messages(messages)


class ChatHistoryStore:
    def __init__(self):
        self._cache = Cache(name="chat_history")

    def __call__(self, user_id: str, conversation_id: str) -> BaseChatMessageHistory:
        # 尝试从缓存中获取历史记录
        key = f"{user_id}-{conversation_id}"
        history: BaseChatMessageHistory = self._cache.get(key)
        if history is not None:
            return history

        # 如果缓存中不存在，则创建新的历史记录并缓存
        sql_user = os.getenv("SQL_USER")
        sql_pwd = os.getenv("SQL_PWD")
        sql_url = os.getenv("SQL_URL")
        sql_db = os.getenv("SQL_DB")
        table_name = os.getenv("SQL_TABLE")
        connection_string = f"postgresql://{sql_user}:{sql_pwd}@{sql_url}/{sql_db}?sslmode=disable"
        sync_connection = psycopg.connect(connection_string)
        # Create the table schema (only needs to be done once)
        PostgresChatMessageHistory.create_tables(sync_connection, table_name)
        history = PgSQLChatMessageHistory(
            table_name,
            conversation_id,
            sync_connection=sync_connection
        )
        self._cache.set(key, history)
        return history

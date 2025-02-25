import os

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI


def create_ai() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo-latest",
        temperature=0.1
    )


def create_ai_with_callbacks(callbacks: list[BaseCallbackHandler]) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo-latest",
        callbacks=callbacks
    )

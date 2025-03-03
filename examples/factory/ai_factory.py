import os

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI


def create_ai() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen2.5-14b-instruct-1m",
        # model="qwen-turbo-latest",
        temperature=0.0
    )

# def create_ai() -> ChatOpenAI:
#     return ChatOpenAI(
#         api_key=os.getenv("HUNYUAN_API_KEY"),
#         base_url="https://api.lkeap.cloud.tencent.com/v1",
#         model="deepseek-r1",
#         temperature=0.0
#     )


def create_ai_with_callbacks(callbacks: list[BaseCallbackHandler]) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo-latest",
        callbacks=callbacks
    )

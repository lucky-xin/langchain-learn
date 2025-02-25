import os

from langchain_openai import ChatOpenAI


def create_ai() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo-latest",
    )
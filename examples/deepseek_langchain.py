# pip3 install langchain_openai
# python3 deepseek_langchain.py
from langchain_openai.chat_models.base import BaseChatOpenAI

llm = BaseChatOpenAI(
    model='deepseek-chat',
    openai_api_key='sk-91502b8e4e3f457a93e905e606efb772',
    openai_api_base='https://api.deepseek.com',
    max_tokens=1024
)

response = llm.invoke("Hi!")
print(response.content)
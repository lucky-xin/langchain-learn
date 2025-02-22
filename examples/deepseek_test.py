from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import OpenAI

from openai import OpenAI

client = OpenAI(
    base_url='https://api.deepseek.com'
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)




# # 定义 ChatPromptTemplate
# template = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful AI bot. Your name is {name}."),
#     ("human", "Hello, how are you doing?"),
#     ("ai", "I'm doing well, thanks!"),
#     ("human", "{user_input}"),
# ])
#
# # 创建链式调用
# chain = template | llm
#
# # 格式化输入
# input_data = {"name": "Bob", "user_input": "给我讲一个埃及法老的笑话"}
#
# # 调用链并获取结果
# result = chain.invoke(input_data)
# print(result)

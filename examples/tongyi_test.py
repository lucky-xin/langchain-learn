from langchain_community.llms import Tongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = Tongyi()

# 定义 ChatPromptTemplate
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

# 创建链式调用
chain = template | llm | StrOutputParser

# 格式化输入
input_data = {"name": "Bob", "user_input": "给我讲一个埃及法老的笑话"}

# 调用链并获取结果
result = chain.invoke(input_data)
print(result)

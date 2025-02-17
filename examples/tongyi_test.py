from langchain.llms import Tongyi
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

template = """
Question: {question}
Answer: 可以逐步思考并详细回答这个问题
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question"]
)

print(prompt)

llm = Tongyi()

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "怎样学习大模型并将其应用到自己的业务中？"

res = llm_chain.run(question)

print(res)
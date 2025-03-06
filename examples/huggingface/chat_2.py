from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-R1",
    max_new_tokens=10,
    cache=False,
    seed=123,
    huggingfacehub_api_token=""
)

chat_model = ChatHuggingFace(llm=llm)

resp = chat_model.invoke("你好")
print(resp)
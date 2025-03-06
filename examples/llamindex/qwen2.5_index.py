import os

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.embeddings.dashscope import DashScopeEmbedding
# Set prompt template for generation (optional)
from llama_index.llms.openai_like import OpenAILike


# conda create -n ai python=3.12
def completion_to_prompt(completion):
    return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"


def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

    if not prompt.startswith("<|im_start|>system"):
        prompt = "<|im_start|>system\n" + prompt

    prompt = prompt + "<|im_start|>assistant\n"

    return prompt


# Set Qwen2.5 as the language model and set generation config
Settings.llm = OpenAILike(
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-turbo-latest",
    is_chat_model=True,
    context_window=30000,
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
)

# LlamaIndex默认使用的Embedding模型被替换为百炼的Embedding模型
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v2"
)

# Set the size of the text chunk for retrieval
Settings.transformations = [SentenceSplitter(chunk_size=1024)]

# 读取被解析的文件目录下所有文件
documents = SimpleDirectoryReader("/tmp/agent").load_data()

# from_documents方法包含对文档进行切片与建立索引两个步骤
index = VectorStoreIndex.from_documents(
    documents=documents,
    # 指定embedding 模型
    embed_model=Settings.embed_model
)

query_engine = index.as_query_engine(Settings.llm)

resp = query_engine.query("TOGAF架构内容框架")

hyde = HyDEQueryTransform(include_original=True)
lyft_hyde_query_engine = TransformQueryEngine(query_engine, hyde)

base_chat_engine = index.as_chat_engine(
    context_window=1024,
    llm=Settings.llm,
    chat_mode=ChatMode.OPENAI,
    system_prompt="""
    你是一个AI助手，你需要根据用户问题，从提供的文档中检索出最相关的内容，并给出最详细的答案。
    如果你不知道答案，请直接返回“抱歉，这个问题我还不知道。”作为答案。
    """
)

query_resp = base_chat_engine.chat("请给我一个关于如何使用LlamaIndex的示例")
print(query_resp)

print("已使用DashScopeEmbedding模型构建了多个文档的向量化索引")

# 输出建立好的索引和压缩好的向量示例
print("输出向量化示例：")
for i, uuid in enumerate(index.vector_store.data.metadata_dict.keys()):
    print("文件名：", end='')
    print(index.vector_store.data.metadata_dict[uuid]['file_name'], end='')
    print("，文件大小：", index.vector_store.data.metadata_dict[uuid]['file_size'], end='')
    print("，文件类型：", index.vector_store.data.metadata_dict[uuid]['file_type'])
    print("压缩后向量：", end='')
    print(index.vector_store.data.embedding_dict[uuid][:3], '\n')

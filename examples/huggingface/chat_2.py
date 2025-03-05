from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint
from transformers.utils import quantization_config

llm_10 = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-70B-Instruct",
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    max_new_tokens=10,
    cache=False,
    seed=123,
)

chat_model = ChatHuggingFace(llm=llm)

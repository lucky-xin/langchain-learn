from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers.utils import quantization_config

# 使用 HuggingFacePipeline 时，模型是加载至本机并在本机运行的
llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "top_k": 50,
        "temperature": 0.1,
    },
)
resp = llm.invoke("Hugging Face is")
print(type(resp))
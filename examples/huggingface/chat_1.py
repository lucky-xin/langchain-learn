from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers.utils import quantization_config

llm = HuggingFacePipeline.from_model_id(
    model_id="ali-vilab/MS-Image2Video",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    model_kwargs={"quantization_config": quantization_config},
)

chat_model = ChatHuggingFace(llm=llm)
response = chat_model.invoke("Generate a video about a cat.")
response.pretty_print()

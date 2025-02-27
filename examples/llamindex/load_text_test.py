import os

from llama_index.core import SimpleDirectoryReader, download_loader, VectorStoreIndex, ServiceContext, Settings
from llama_index.llms.openai_like import OpenAILike
from openai.types.beta import VectorStore

documents = SimpleDirectoryReader('data').load_data()

index = VectorStoreIndex.from_documents(documents)


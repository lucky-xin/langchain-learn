from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

documents = SimpleDirectoryReader('data').load_data()

index = VectorStoreIndex.from_documents(documents)


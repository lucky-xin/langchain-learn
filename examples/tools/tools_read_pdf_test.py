from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

embeddings = DashScopeEmbeddings()

loader = PyPDFLoader("/tmp/agent/清华大学-DeepSeek从入门到精通2025.pdf")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
texts = text_splitter.split_documents(docs)
db = Chroma.from_documents(texts, embeddings)

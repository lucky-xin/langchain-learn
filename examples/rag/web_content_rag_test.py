import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

loader = WebBaseLoader(
    web_path='https://lilianweng.github.io/posts/2023-06-23-agent/',
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('post-header', 'post-title', 'post-content'))
    )
)

docs = loader.load()

text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=2000, chunk_overlap=100)
split_documents = text_splitter.split_documents(docs)

for i, d in enumerate(split_documents):
    print(f"{i}: {d.page_content}")
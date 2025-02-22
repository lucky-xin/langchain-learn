import os

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.document_loaders import BaseLoader
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from examples.factory.ai_factory import create_ai

# Customize the layout
st.set_page_config(page_title="Local AI Chat Powered by Xinference", page_icon="🤖", layout="wide")


# Write uploaded file in temp dir
def write_text_file(content, file_path: str):
    try:
        with open(file_path, 'wb') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
        return False


# Prepare prompt template
prompt_template = """
使用下面的上下文来回答问题。
如果你不知道答案，就说你不知道，不要编造答案。
{context}

问题: {question}

回答:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize the Xinference LLM & Embeddings
embeddings = DashScopeEmbeddings()

llm = create_ai()

st.title("📄文档对话")
uploaded_file = st.file_uploader("上传文件", type=["txt", "pdf"])


def create_vector_store() -> VectorStore:
    return InMemoryVectorStore(embeddings)


if uploaded_file is not None:
    file_type = uploaded_file.type
    content = uploaded_file.getvalue()

    parent_path = "/tmp/agent"
    file_path = os.path.join(parent_path, uploaded_file.name)
    write_text_file(content, file_path)
    loader: BaseLoader = None
    print(f"文件类型：{file_type}")
    if file_type == "application/pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "txt":
        loader = TextLoader(file_path)
    else:
        st.error("不支持的文件类型")
        exit(1)

    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)
    vector_store = create_vector_store()
    db = vector_store.from_documents(texts, embeddings)
    st.success("上传文档成功")

    # Query through LLM
    question = st.text_input("提问", placeholder="请问我任何关于文章的问题", disabled=not uploaded_file)
    if question:
        similar_doc = db.similarity_search(question, k=1)
        st.write("相关上下文：")
        st.write(similar_doc)
        context = similar_doc[0].page_content
        query_llm = prompt | llm
        response = query_llm.invoke({"context": context, "question": question})
        st.write(f"回答：{response.content}")

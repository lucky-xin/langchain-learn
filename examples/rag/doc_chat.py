import os
from typing import List

import streamlit as st
from cache3 import Cache
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from pydantic import BaseModel, Field

from examples.factory.ai_factory import create_ai


class ChatResp(BaseModel):
    answer: str = Field(description="回答")
    context: List[str] = Field(description="上下文")


def create_vector_store() -> VectorStore:
    # return Chroma("langchain_store", DashScopeEmbeddings())
    return InMemoryVectorStore(DashScopeEmbeddings())


# Write uploaded file in temp dir
def write_file(fp: str, content):
    try:
        with open(fp, 'wb') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
        return False


def create_prompt() -> BasePromptTemplate:
    # Prepare prompt template
    prompt_template = """
使用下面的上下文来回答问题。
如果你不知道答案，就说你不知道，不要编造答案。
{context}

问题: 
{question}

"""
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])


# loading PDF, DOCX and TXT files as LangChain Documents
def load_documents(file) -> list[Document]:
    _, extension = os.path.splitext(file)
    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return []
    return loader.load()


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20) -> List[Document]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def add_documents(store: VectorStore, docs: List[Document]):
    # if you want to use a specific directory for chromadb
    # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
    store.add_documents(docs)


# clear the chat history from streamlit session state
def clear_history():
    pass
    # if 'history' in st.session_state:
    # del st.session_state['history']


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004


def create_tmp_dir(tmp_dir: str):
    os.makedirs(tmp_dir, exist_ok=True)


if __name__ == "__main__":
    llm = create_ai()
    cache = Cache(name="uploaded_file")
    # See full prompt at https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    parent_path = "/tmp/agent"
    create_tmp_dir(parent_path)

    # st.image('img.png')
    st.subheader('Qwen🤖')
    st.session_state.vs = create_vector_store()

    with st.sidebar:

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        chunks = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:  # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):
                # writing the file from RAM to the current directory on disk
                file_path = os.path.join(parent_path, uploaded_file.name)
                print(f'Writing {uploaded_file.name} to {file_path}')
                write_file(file_path, uploaded_file.read())

                documents = load_documents(file_path)
                chunk_docs = chunk_data(documents, chunk_size=chunks)
                st.session_state.vs.add_documents(chunk_docs)

                st.write(f'Chunk size: {chunks}, Chunks: {len(chunk_docs)}')
                tokens, embedding_cost = calculate_embedding_cost(chunk_docs)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')
                st.success('File uploaded, chunked and embedded successfully.')

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        st.chat_message(msg["role"])
        st.markdown(msg["content"])
    parser = JsonOutputParser(pydantic_object=ChatResp)
    rag_chain = create_retrieval_chain(st.session_state.vs.as_retriever(), combine_docs_chain)
    q = st.chat_input(placeholder="请问我任何关于文章的问题", disabled=not uploaded_file)
    if q:
        st.chat_message("user").markdown(q)
        st.session_state.messages.append({"role": "user", "content": q})
        output_placeholder = st.empty()
        collected_messages = ""
        with st.chat_message("assistant"):
            # st_callback = StreamlitCallbackHandler(st.container())
            # stream = rag_chain.stream({"input": q}, config={"callbacks": [st_callback]})
            # st.write(stream)
            stream = rag_chain.stream({"input": q})
            for chunk in stream:
                if "answer" in chunk:
                    collected_messages += chunk.get("answer")
                    output_placeholder.text(collected_messages)
            st.session_state.messages.append({"role": "assistant", "content": collected_messages})

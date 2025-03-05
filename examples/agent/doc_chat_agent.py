import os
from typing import List, Iterable

import streamlit as st
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from examples.factory.ai_factory import create_ai


def create_vector_store() -> VectorStore:
    # return Chroma("langchain_store", DashScopeEmbeddings())
    return InMemoryVectorStore(DashScopeEmbeddings())


# Write uploaded file in temp dir
def write_file(fp: str, content: bytes):
    try:
        with open(fp, 'wb') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
        return False


# loading PDF, DOCX and TXT files as LangChain Documents
def load_documents(file: str) -> list[Document]:
    _, extension = os.path.splitext(file)
    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return []
    return loader.load()


# splitting data in chunks
def split_documents(data: Iterable[Document], chunk_size=2000, chunk_overlap=200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)

# clear the chat history from streamlit session state
def clear_history():
    pass
    # if 'history' in st.session_state:
    # del st.session_state['history']


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts: Iterable[Document]):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004


def create_tmp_dir(tmp_dir: str):
    os.makedirs(tmp_dir, exist_ok=True)


@st.cache_resource(ttl="1d")
def create_prompt() -> BasePromptTemplate:
    # 指令模板
    instructions = """你是一个设计用于査询文档来回答问题的代理您可以使用文档检索工具，
    并优先基于检索内容来回答问题，如果从文档中找不到任何信息用于回答问题，可以通过其他工具搜索答案，如果所有的工具都不能找到答案，则只需返回“抱歉，这个问题我还不知道。”作为答案。
    你需要以JSON结构返回。JSON结构体包含output字段，output是你给用户返回的内容。
    """
    base_prompt = hub.pull("hwchase17/react")
    return base_prompt.partial(instructions=instructions)


def create_agent(retriever: BaseRetriever) -> AgentExecutor:
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
        chat_memory=msgs
    )
    tool = create_retriever_tool(
        retriever=retriever,
        name="文档检索",
        description="用于检索用户提出的问题，并基于检索到的文档内容进行回复"
    )

    tools = [
        tool,
        WikipediaQueryRun(
            name="wiki-tool",
            description="look up things in wikipedia",
            api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100),
        ),
        DuckDuckGoSearchResults()
    ]
    react_agent = create_react_agent(create_ai(), tools, create_prompt())
    return AgentExecutor(
        agent=react_agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors="没有从知识库检索到相似的内容"
    )


def init():
    if "vs" not in st.session_state:
        st.session_state.vs = create_vector_store()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

def upload_file(uf: UploadedFile, chunk_size: int):
    parent_path = "/tmp/agent"
    create_tmp_dir(parent_path)
    # writing the file from RAM to the current directory on disk
    file_path = os.path.join(parent_path, uf.name)
    print(f'Writing {uf.name} to {file_path}')
    write_file(file_path, uf.read())
    documents = load_documents(file_path)
    chunk_docs = split_documents(documents, chunk_size=chunk_size)
    st.session_state.vs.add_documents(chunk_docs)
    st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunk_docs)}')
    tokens, embedding_cost = calculate_embedding_cost(chunk_docs)
    st.write(f"Embedding cost: {embedding_cost:.4f}, total tokens:{tokens}")
    st.success('File uploaded, chunked and embedded successfully.')


if __name__ == "__main__":
    init()
    # st.image('img.png')
    st.subheader('Qwen🤖')
    with st.sidebar:
        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        # chunk size number widget
        chunks = st.number_input('Chunk size:', min_value=2000, max_value=4000, value=2000, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)
        if uploaded_file and add_data:  # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):
                upload_file(uploaded_file, chunks)

    q = st.chat_input(placeholder="请问我任何关于文章的问题")
    if q:
        st.chat_message("user").markdown(q)
        st.session_state.messages.append({"role": "user", "content": q})
        collected_messages = ""
        with st.chat_message("assistant"):
            output_placeholder = st.empty()
            st_cb = StreamlitCallbackHandler(st.container())
            agent = create_agent(st.session_state.vs.as_retriever())
            stream = agent.stream({"input": q}, config={"callbacks": [st_cb]})
            for chunk in stream:
                if "output" in chunk:
                    collected_messages += chunk.get("output")
                    output_placeholder.markdown(collected_messages + "▌")
            output_placeholder.markdown(collected_messages)
            st.session_state.messages.append({"role": "assistant", "content": collected_messages})

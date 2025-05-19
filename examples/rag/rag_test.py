import os.path
import tempfile

import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.tools.retriever import create_retriever_tool
from langchain_community.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from examples.factory.llm import LLMType, LLMFactory


@st.cache_resource(ttl="1h")
def configure_retriever(files):
    docs = []
    tmp_dir = tempfile.TemporaryDirectory(dir="/tmp")
    for file in files:
        temp_fp = os.path.join(tmp_dir.name, file.name)
        with open(temp_fp, "wb") as f:
            f.write(file.getvalue())
        loader = TextLoader(temp_fp, encoding="utf-8")
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = DashScopeEmbeddings()
    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    return vector_db.as_retriever()

st.title("文档问答")
st.set_page_config(
    page_title="文档问答",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


upload_files = st.sidebar.file_uploader("上传文件", type=["pdf", "docx", "md", "txt"], accept_multiple_files=True)

if not upload_files:
    st.info("请先上传文件")
    st.stop()

retriever = configure_retriever(upload_files)

if "messages" not in st.session_state or not st.sidebar.button("清空聊天记录"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好，我是一个智能文档助理机器人，你可以向我提问任何问题。"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

tool = create_retriever_tool(retriever, "文档检索", "用于检索用户提出的问题，并基于检索到的文档内容进行回复")
tools = [tool]

# 指令模板
instructions = """您是一个设计用于査询文档来回答问题的代理您可以使用文档检索工具，
并基于检索内容来回答问题您可能不查询文档就知道答案，但是您仍然应该查询文档来获得答案。
如果您从文档中找不到任何信息用于回答问题，则只需返回“抱歉，这个问题我还不知道。”作为答案。
"""

# 基础提示模板
base_prompt_template = """
{instructions}

TOOLS:
--------
You have access to the following tools:

{tools}

To use a tool, please use the following format:

ZWJ```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: {input}
Observation: the result of the action
ZWJ```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

ZWJ```
Thought: Do I need to use a tool? 
NoFinal Answer: [your response here]
ZWJ```

Begin!

Previous conversation history:
{chat history}

New input: {input}
{agent_scratchpad}
"""


base_prompt = PromptTemplate.from_template(base_prompt_template)

prompt = base_prompt.partial(instructions=instructions)
llm_factory = LLMFactory(
    llm_type=LLMType.LLM_TYPE_QWENAI,
)

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output",
    chat_memory=msgs
)
agent = create_react_agent(llm_factory.create_llm(), tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors="没有从知识库检索到相似的内容"
)

user_query = st.chat_input(placeholder="请输入问题")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        st.write("思考中...")
        st_cb = StreamlitCallbackHandler(st.container())
        config = {"callbacks": [st_cb]}
        response = agent_executor.invoke({"input": user_query}, config=config)
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        st.write(response["output"])

# app = FastAPI(title="我的LangChain服务", version="1.0.0", description="LangChain服务")
#
# # add_routes(
# #     app,
# #     chain,
# #     path="/chain"
# # )
#
# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=8000)

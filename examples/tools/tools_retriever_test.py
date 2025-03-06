import uuid

from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from examples.factory.ai_factory import create_chat_ai
from examples.his.chat_history_store import ChatHistoryStore

loader = WebBaseLoader("https://zh.wikipedia.org/wiki/%E7%8C%AB")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
).split_documents(docs)

vector = FAISS.from_documents(documents, DashScopeEmbeddings())
retriever = vector.as_retriever()

print(retriever.invoke("猫的特征")[0])

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="wiki_search",
    description="""
    A retriever that can be used to retrieve information about cats.
    """,
)

llm = create_chat_ai()
search = TavilySearchResults()
tools = [retriever_tool, search]

prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

session_id = str(uuid.uuid4())
user_id = str(uuid.uuid4())

input_data = {
    "name": "Bob",
    "question": "世界上最大的猫是什么品种",
    "user_id": user_id
}
config = {
    'configurable': {
        'conversation_id': session_id,
        'user_id': user_id
    }
}

chain_with_history = RunnableWithMessageHistory(
    agent_executor,
    ChatHistoryStore(),
    input_messages_key='question',
    history_messages_key='history',
    history_factory_config=[
        ConfigurableFieldSpec(
            id='user_id',
            name="User ID",
            annotation=str,
            description='用户的唯一标识符',
            default="none",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id='conversation_id',
            name="Conversation ID",
            annotation=str,
            description='对话的唯一标识符',
            default="none",
            is_shared=True,
        )
    ]
)

res = chain_with_history.invoke(
    input=input_data,
    config=config
)
print(res)

from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from examples.factory.ai_factory import create_chat_ai

hyde_prompt_template = """
get the guidelines to this requirement {input}. 
Use the {guideline_name} which are in the context and 
think how these guidelines will be helpful to this requirement. print 
only the final output.
"""
prompt = PromptTemplate.from_template(hyde_prompt_template)

embeddings = DashScopeEmbeddings(model="text-embedding-v2")
llm_chain = LLMChain(llm=create_chat_ai(), prompt=prompt)

hyde_embeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=embeddings
)

faiss = FAISS.load_local(
    folder_path="local path",
    embeddings=hyde_embeddings,
    allow_dangerous_deserialization=True
)

retriever = faiss.as_retriever()
q = ""
resp = retriever.invoke(q)
print(resp)

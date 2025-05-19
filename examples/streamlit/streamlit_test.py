import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from examples.factory.llm import LLMFactory, LLMType

st_callback = StreamlitCallbackHandler(st.container())

llm_factory = LLMFactory(
    llm_type=LLMType.LLM_TYPE_QWENAI,
)

tools = load_tools(["ddg-search"])
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm_factory.create_llm(), tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
        )
        st.write(response["output"])

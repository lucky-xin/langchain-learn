import concurrent.futures

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Customize the layout
st.set_page_config(page_title="Local AI Chat Powered by Xinference", page_icon="🤖", layout="wide")


# Write uploaded file in temp dir
def write_text_file(content, file_path):
    try:
        with open(file_path, 'w') as file:
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
回答:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize the Xinference LLM & Embeddings
embeddings = DashScopeEmbeddings()
llm = ChatTongyi(max_retries=5)

st.title("📄文档对话")
uploaded_file = st.file_uploader("上传文件", type="txt")

if uploaded_file is not None:
    content = uploaded_file.read().decode('utf-8')
    file_path = "/tmp/file.txt"
    write_text_file(content, file_path)

    loader = TextLoader(file_path)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)
    db = Chroma.from_documents(texts, embeddings)
    st.success("上传文档成功")

    # Query through LLM
    question = st.text_input("提问", placeholder="请问我任何关于文章的问题", disabled=not uploaded_file)
    if question:
        similar_doc = db.similarity_search(question, k=1)
        st.write("相关上下文：")
        st.write(similar_doc)
        context = similar_doc[0].page_content
        query_llm = llm | prompt
        response = query_llm.run({"context": context, "question": question})
        st.write(f"回答：{response}")

ctx = get_script_run_ctx()
##Some code##


if __name__ == '__main__':
    if st.button('run'):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = executor.map(lambda f: multip(*f), final)

            for t in executor._threads:
                add_script_run_ctx(thread=t, ctx=ctx)

        res = []
        for value in results:
            res.append(value)
        st.write(f'res: {res}')

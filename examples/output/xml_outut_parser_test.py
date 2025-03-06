from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import PromptTemplate

from examples.factory.ai_factory import create_chat_ai

if __name__ == '__main__':
    jq = "生成周星驰的简化电影作品列表，按照最新的时间降序"
    parser = XMLOutputParser()
    prompt = PromptTemplate(
        template="回答用户查询。\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    print(parser.get_format_instructions())
    llm = create_chat_ai()
    chain = prompt | llm
    resp = chain.invoke({"query": jq})
    xml_output = parser.parse(resp)
    print(xml_output)

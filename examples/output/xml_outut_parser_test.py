from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import PromptTemplate

from examples.factory.llm import LLMFactory, LLMType

if __name__ == '__main__':
    jq = "生成周星驰的简化电影作品列表，按照最新的时间降序"
    parser = XMLOutputParser()
    prompt = PromptTemplate(
        template="回答用户查询。\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    print(parser.get_format_instructions())
    llm_factory = LLMFactory(
        llm_type=LLMType.LLM_TYPE_QWENAI,
    )
    chain = prompt | llm_factory.create_llm()
    resp = chain.invoke({"query": jq})
    xml_output = parser.parse(resp)
    print(xml_output)

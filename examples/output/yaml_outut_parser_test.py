from langchain.output_parsers import YamlOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from examples.factory.ai_factory import create_chat_ai


class Joke(BaseModel):
    setup: str = Field(description="设置笑话的问题")
    punchline: str = Field(description="解决笑话的答案")


if __name__ == '__main__':
    jq = "告诉我一个笑话"
    parser = YamlOutputParser(pydantic_object=Joke)
    prompt = PromptTemplate(
        template="回答用户查询。\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = create_chat_ai()
    chain = prompt | llm
    resp = chain.invoke({"query": jq})
    print(resp)

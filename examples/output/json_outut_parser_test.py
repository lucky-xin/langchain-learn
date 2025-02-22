from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from examples.factory.ai_factory import create_ai


class Joke(BaseModel):
    setup: str = Field(description="设置笑话的问题")
    punchline: str = Field(description="解决笑话的答案")


if __name__ == '__main__':

    jq = "告诉我一个笑话"
    parser = JsonOutputParser(pydantic_object=Joke)
    prompt = PromptTemplate(
        template="回答用户查询。\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = create_ai()
    chain = prompt | llm | parser
    for s in chain.stream({"query": jq}):
        print(s)

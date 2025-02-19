import base64
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_community.llms import Tongyi
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec

from examples.his.chat_history_store import ChatHistoryStore


def get_current_timestamp() -> int:
    # 获取当前时间戳（带时区）
    timezone = ZoneInfo("Asia/Shanghai")  # 例如，上海时区
    current_time = datetime.now(timezone)
    return int(current_time.timestamp())


if __name__ == '__main__':
    llm = Tongyi()
    additionals = {"timestamp": "{timestamp}", "user_id": "{user_id}"}
    fp = "image.jpg"
    # 读取图片并转换为base64编码
    with open(fp, "rb") as image_file:
        img_b64 = base64.b64encode(image_file.read()).decode('utf-8')
    # 定义 ChatPromptTemplate
    template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are a helpful AI bot. Your name is {name}.",
            additional_kwargs=additionals
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": "用中文描述这张图片的天气"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{img_b64}"}}
            ],
            additional_kwargs=additionals
        ),
        AIMessage(
            content="I'm doing well, thanks!",
            additional_kwargs=additionals
        ),
        # 历史消息占位符
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(
            content="{question}",
            additional_kwargs=additionals
        )
    ])

    # 创建链式调用
    chain = template | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        ChatHistoryStore(),
        # lambda conversation_id, user_id: InMemoryChatMessageHistory(),
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

    session_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    input_data = {
        "name": "Bob",
        "question": "给我讲一个埃及法老的笑话",
        "timestamp": get_current_timestamp(),
        "user_id": user_id
    }
    config = {
        'configurable': {
            'conversation_id': session_id,
            'user_id': user_id,
            "timestamp": get_current_timestamp(),
        }
    }
    # 调用链并获取结果
    res = chain_with_history.invoke(
        input=input_data,
        config=config
    )
    print(res)
    input_data["timestamp"] = get_current_timestamp(),
    input_data["question"] = "TOGAF 10和TOGAF 9.2有什么不一样"
    res = chain_with_history.invoke(
        input=input_data,
        config=config
    )
    print(res)

    input_data["timestamp"] = get_current_timestamp(),
    input_data["question"] = "什么？"
    res = chain_with_history.invoke(
        input=input_data,
        config=config
    )
    print(res)

    input_data["timestamp"] = get_current_timestamp(),
    config["configurable"]["conversation_id"] = str(uuid.uuid4())
    res = chain_with_history.invoke(
        input=input_data,
        config=config
    )
    print(res)

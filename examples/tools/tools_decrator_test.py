from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class CalaInput(BaseModel):
    a: int = Field(description="The first number.")
    b: int = Field(description="The second number.")


@tool
def get_current_timestamp() -> int:
    """Get the current timestamp in milliseconds."""
    # 获取当前时间戳（带时区）
    timezone = ZoneInfo("Asia/Shanghai")  # 例如，上海时区
    current_time = datetime.now(timezone)
    return int(current_time.timestamp())


@tool
def multiply_1(a: int, b: int) -> int:
    """Multiply two numbers."""
    return (a * b)


@tool("multiplication-tool", args_schema=CalaInput, return_direct=True)
def multiply_2(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


if __name__ == '__main__':
    print(multiply_2.name)
    print(multiply_2.description)
    print(multiply_2.args)

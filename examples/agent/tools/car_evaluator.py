from typing import Optional, Type

import requests.auth
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from requests.exceptions import HTTPError, ReadTimeout
from urllib3.exceptions import ConnectionError


class CarEvaluateInput(BaseModel):
    """Input for the CarEvaluateTool tool."""

    trim_id: str = Field(description="车型号id", pattern="^tri(\d+)$")
    city_id: str = Field(description="城市id", pattern="^cit(\d+)$")
    color_id: str = Field(default="Col09", description="颜色id", pattern="^Col(\d+)$")
    mileage: float = Field(description="行驶里程，单位为万公里，必须大于0，如：2.4，表示2.4万公里")
    reg_time: str = Field(description="车上牌时间，格式为yyyyMMdd,如：20210401")


class CarEvaluateTool(BaseTool):
    """二手车精准估值工具"""

    name: str = "car_evaluate"

    description: str = """
    二手车估值API接口基于精准数据重塑行业标准，
    通过对车辆外观、车型号、上牌时间、行驶里程、以及上牌城市等多个维度进行综合评估，
    从而得出一辆二手车的精准估值
    """

    args_schema: Type[BaseModel] = CarEvaluateInput

    """服务端点"""
    endpoint: str = "https://openapi.pistonint.com/evaluate"

    """认证器"""
    auth: requests.auth.AuthBase

    def _run(
            self,
            trim_id: str,
            city_id: str,
            color_id: str,
            mileage: float,
            reg_time: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Use the Car Evaluate tool.
        Args:
             trim_id: 车型号id
             city_id: 城市id
             color_id: 颜色id
             mileage:  行驶里程，单位为万公里，必须大于0，如：2.4，表示2.4万公里
             reg_time: 车上牌时间，格式为yyyyMMdd,如：20210401
             run_manager: Optional callback manager.
        Returns: Formatted news results or error message.
        """

        try:
            resp = requests.post(
                url=self.endpoint,
                json={
                    "datas": [
                        {
                            "trimId": trim_id,
                            "cityId": city_id,
                            "colorId": color_id,
                            "mileage": mileage,
                            "regTime": reg_time
                        }
                    ]
                },
                auth=self.auth,
                headers={
                    "Content-Type": "application/json"
                },
            ).json()
            if resp.get("code") == 0:
                return resp.get("msg", "evaluate failed")
            vals = resp.get("data", [])
            data = vals[0] if vals else {}
            sell = data.get("sell", {})
            buy = data.get("buy", {})
            return f"""
            指导价：{data.get("msrp", 0)}
            卖车价：
                车况等级A（优秀车况）估值： {sell.get("valueA", 0)}（保值率：{sell.get("valuePctA", 0.0) * 100}%）
                车况等级B（良好车况）估值： {sell.get("valueB", 0)}（保值率：{sell.get("valuePctB", 0.0) * 100}%）
                车况等级C（一般车况）估值： {sell.get("valueC", 0)}（保值率：{sell.get("valuePctC", 0.0) * 100}%）
            收车价：
                车况等级A（优秀车况）估值： {buy.get("valueA", 0)}（保值率：{buy.get("valuePctA", 0.0) * 100}%）
                车况等级B（良好车况）估值： {buy.get("valueB", 0)}（保值率：{buy.get("valuePctB", 0.0) * 100}%）
                车况等级C（一般车况）估值： {buy.get("valueC", 0)}（保值率：{buy.get("valuePctC", 0.0) * 100}%）
            """

        except (HTTPError, ReadTimeout, ConnectionError):
            return "execute HTTP request failed"

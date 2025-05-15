from typing import Any

import httpx
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport

# Initialize FastMCP server
weather_mcp = FastMCP("weather")

app = Server("mcp-website-fetcher")  # 创建MCP服务器实例，名称为"mcp-website-fetcher"
sse = SseServerTransport("/messages/")  # 创建SSE服务器传输实例，路径为"/messages/"

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return None


def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""


@weather_mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.
    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)
    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."
    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


@weather_mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)
    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)
    if not forecast_data:
        return "Unable to fetch detailed forecast."
    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forecasts.append(forecast)
    return "\n---\n".join(forecasts)


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    # 定义异步函数list_tools，用于列出可用的工具
    # 返回: Tool对象列表，描述可用工具

    return [
        types.Tool(
            name="forecast",
            description="Get weather forecast for a location",
            inputSchema={
                "type": "object",
                "required": ["latitude", "longitude"],
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude of the location",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude of the location",
                    }
                },
            },
        ),
        types.Tool(
            name="alerts",
            description="Get weather alerts for a US state",
            inputSchema={
                "type": "object",
                "required": ["state"],
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The keyword you want to search",
                    }
                },
            },
        ),
    ]


async def handle_sse(req):
    # 定义异步函数handle_sse，处理SSE请求
    # 参数: request - HTTP请求对象

    async with sse.connect_sse(
            req.scope, req.receive, req._send
    ) as streams:
        # 建立SSE连接，获取输入输出流
        await app.run(
            streams[0], streams[1], app.create_initialization_options()
        )  # 运行MCP应用，处理SSE连接


from starlette.applications import Starlette
from starlette.routing import Mount, Route

starlette_app = Starlette(
    debug=True,  # 启用调试模式
    routes=[
        Route("/sse", endpoint=handle_sse),  # 设置/sse路由，处理函数为handle_sse
        Mount("/messages/", app=sse.handle_post_message),  # 挂载/messages/路径，处理POST消息
    ],
)  # 创建Starlette应用实例，配置路由


if __name__ == "__main__":
    import uvicorn  # 导入uvicorn ASGI服务器

    uvicorn.run(starlette_app, host="0.0.0.0", port=8888)  # 运行Starlette应用，监听127.0.0.1和指定端口

import httpx
import mcp.types as types
from langchain_community.tools import DuckDuckGoSearchResults
from mcp.server.lowlevel import Server
from mcp.types import TextContent


async def fetch_website(
        url: str,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    # 定义异步函数fetch_website，用于获取网页内容
    # 参数: url - 要获取内容的网站URL
    # 返回: 包含文本、图像或嵌入资源的列表
    headers = {
        "User-Agent": "MCP Test Server (github.com/modelcontextprotocol/python-sdk)"
    }  # 设置HTTP请求头，指定User-Agent

    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        # 创建异步HTTP客户端，设置自动跟随重定向并使用上面定义的headers
        response = await client.get(url)  # 异步发送GET请求获取URL内容
        response.raise_for_status()  # 如果HTTP响应状态码不是2xx，则抛出异常
        return [types.TextContent(type="text", text=response.text)]  # 返回网页文本内容作为TextContent对象


async def duck_duck_go_search(keyword: str, page_num: int = 2) -> list[TextContent]:
    # 定义异步函数search_by_bing，使用Bing搜索并提取网页内容
    # 参数: searchKeyWords - 搜索关键词, page_num - 要检索的页面数量，默认为2
    search_duck_duck_go = DuckDuckGoSearchResults(num_results=page_num)
    resp = search_duck_duck_go.run(keyword)
    return [types.TextContent(type="text", text=str(resp))]  # 返回包含所有搜索结果的TextContent对象


## 创建SSE Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route

sse = SseServerTransport("/messages/")  # 创建SSE服务器传输实例，路径为"/messages/"
app = Server("mcp-website-fetcher")  # 创建MCP服务器实例，名称为"mcp-website-fetcher"


@app.call_tool()
async def fetch_tool(
        name: str,
        arguments: dict) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    # 定义异步函数fetch_tool，作为MCP工具调用处理器
    # 参数: name - 工具名称, arguments - 工具参数字典
    # 返回: 包含文本、图像或嵌入资源的列表

    if name == "fetch":
        # 如果调用的是"fetch"工具
        if "url" not in arguments:
            raise ValueError("Missing required argument 'url'")  # 检查是否提供了必需的url参数
        return await fetch_website(arguments["url"])  # 调用fetch_website函数获取网页内容

    elif name == "searcher":
        # 如果调用的是"searcher"工具
        if "keyword" not in arguments:
            # 检查是否提供了必需的searchKeyWords参数
            raise ValueError("Missing required argument 'keyword'")
            # 调用search_by_bing函数进行搜索
        return await duck_duck_go_search(arguments["keyword"], arguments["page_num"])
    return []


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    # 定义异步函数list_tools，用于列出可用的工具
    # 返回: Tool对象列表，描述可用工具

    return [
        types.Tool(
            name="fetch",  # 工具名称
            description="Fetches a website and returns its content",  # 工具描述
            inputSchema={  # 输入模式定义
                "type": "object",
                "required": ["url"],  # 必需参数
                "properties": {
                    "url": {  # url参数定义
                        "type": "string",
                        "description": "URL to fetch",
                    }
                },
            },
        ),
        types.Tool(
            name="searcher",  # 工具名称
            description="Search and Extract Content From Web Page",  # 工具描述
            inputSchema={  # 输入模式定义
                "type": "object",
                "required": ["keyword"],  # 必需参数
                "properties": {
                    "keyword": {  # searchKeyWords参数定义
                        "type": "string",
                        "description": "The keyword you want to search",
                    },
                    "page_num": {  # page_num参数定义
                        "type": "integer",
                        "default": 2,
                        "description": "Number of pages to retrieve, default is 2",
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


starlette_app = Starlette(
    debug=True,  # 启用调试模式
    routes=[
        Route("/sse", endpoint=handle_sse),  # 设置/sse路由，处理函数为handle_sse
        Mount("/messages/", app=sse.handle_post_message),  # 挂载/messages/路径，处理POST消息
    ],
)  # 创建Starlette应用实例，配置路由

import uvicorn  # 导入uvicorn ASGI服务器

uvicorn.run(starlette_app, host="0.0.0.0", port=8888)  # 运行Starlette应用，监听127.0.0.1和指定端口

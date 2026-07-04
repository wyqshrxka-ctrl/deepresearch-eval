"""
MCP Server with 3 tools: search_web / fetch_url / calculator

启动方式：
    python -m mcp_server.main

被 client 通过 stdio 连接调用。

设计要点：
1. 每个工具用 @server.tool() 装饰器声明
2. 工具描述要详细——LLM 选工具是 retrieval 问题
3. 错误信息要清晰——LLM 看到错误才能自我纠正
"""

import asyncio
import ast
import math
import operator
import os
from typing import Any

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

load_dotenv()

# ============================================================
# Server 实例
# ============================================================
server = Server("deepresearch-tools")


# ============================================================
# Tool 1: search_web —— 完整实现（参考用）
# ============================================================
@server.list_tools()
async def list_tools() -> list[Tool]:
    """声明所有可用工具。"""
    return [
        Tool(
            name="search_web",
            description=(
                "在互联网搜索关键词，返回 top 5 结果（标题 + URL + 摘要）。"
                "适用场景：需要获取最新信息、查询不知道的事实、寻找参考资料。"
                "不适用：搜不到具体网页内容，要看具体页面请用 fetch_url。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，建议 3-10 个词，太短不准、太长会失败",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="fetch_url",
            description=(
                "抓取指定 URL 的网页内容，返回纯文本（去除 HTML 标签）。"
                "适用场景：search_web 找到 URL 后，需要看具体内容。"
                "限制：只抓 HTML，不支持 PDF / 视频 / 大文件（>1MB 自动截断）。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "完整的 http/https URL",
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="calculator",
            description=(
                "计算数学表达式，支持 +-*/、括号、sqrt、log、sin、cos 等基本函数。"
                "适用场景：需要精确计算时，避免 LLM 自己算错。"
                "示例：'(123 + 456) * 7.89'、'sqrt(2)'。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python 表达式，会用 eval 执行（已沙箱）",
                    },
                },
                "required": ["expression"],
            },
        ),
    ]


# ============================================================
# Tool 调用分发
# ============================================================
@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """根据 tool name 分发到具体实现。"""
    if name == "search_web":
        result = await search_web(arguments["query"])
    elif name == "fetch_url":
        result = await fetch_url(arguments["url"])
    elif name == "calculator":
        result = await calculator(arguments["expression"])
    else:
        result = f"❌ 未知工具：{name}"

    return [TextContent(type="text", text=result)]


# ============================================================
# 工具具体实现
# ============================================================

async def search_web(query: str) -> str:
    """
    用 Tavily 搜索 API 实现网络搜索。
    未配置 API key 时用模拟数据兜底。
    """
    api_key = os.getenv("TAVILY_API_KEY")

    # 有真实 API key 时调真实接口
    if api_key and api_key != "tvly-xxx":
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query,
                        "max_results": 5,
                        "search_depth": "basic",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("results"):
                    lines = [f"搜索 '{query}' 的 top 5 结果：\n"]
                    for i, r in enumerate(data["results"][:5], 1):
                        lines.append(
                            f"{i}. {r.get('title', '无标题')}"
                        )
                        lines.append(f"   URL: {r.get('url', '')}")
                        lines.append(
                            f"   摘要: {r.get('content', '')[:200]}"
                        )
                        lines.append("")
                    return "\n".join(lines)
            except httpx.HTTPError:
                pass  # 失败后用模拟数据兜底

    # 模拟数据兜底（无 API key 或真实请求失败时使用）
    # 用关键词生成相关的模拟结果
    mock_results = {
        "天气": (
            f"搜索 '{query}' 的模拟结果：\n"
            f"1. 北京今日天气 - 晴天，气温 25-30°C，湿度 40%\n"
            f"   摘要：北京今日天气晴朗，气温舒适，适合户外活动。\n"
            f"2. 穿搭推荐 - 根据气温选择衣物\n"
            f"   摘要：25-30°C 建议穿短袖，注意防晒。\n"
        ),
    }
    for key, content in mock_results.items():
        if key in query:
            return content

    # 默认：按搜索词生成通用结果
    return (
        f"搜索 '{query}' 的模拟结果：\n"
        f"1. 关于「{query}」的介绍\n"
        f"   URL: https://example.com/intro\n"
        f"   摘要：{query} 是一个重要的概念/技术/话题，广泛应用于相关领域。\n"
        f"2. 「{query}」的详细指南\n"
        f"   URL: https://example.com/guide\n"
        f"   摘要：本文详细介绍 {query} 的核心原理、使用方法和最佳实践。\n"
        f"3. 「{query}」最新动态\n"
        f"   URL: https://example.com/news\n"
        f"   摘要：关于 {query} 的最新进展和行业趋势。\n"
    )


async def fetch_url(url: str) -> str:
    """
    抓取 URL 返回纯文本。

    实现要点：
    - 设 User-Agent 避免被反爬
    - 30s 超时
    - 1MB 内容上限
    - 用 BeautifulSoup 提取纯文本
    """
    async with httpx.AsyncClient(
        timeout=30.0,
        headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearchBot/0.1)"},
        follow_redirects=True,
    ) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return f"❌ 抓取失败：{e}"

        # 大小限制
        if len(resp.content) > 1024 * 1024:
            return "❌ 网页过大（>1MB），无法抓取"

        # 提取纯文本
        soup = BeautifulSoup(resp.text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator="\n", strip=True)

        # 截断到 5000 字符（避免 LLM context 爆炸）
        if len(text) > 5000:
            text = text[:5000] + "\n\n... [内容已截断]"

        return f"URL: {url}\n标题: {soup.title.string if soup.title else '无'}\n\n内容：\n{text}"


async def calculator(expression: str) -> str:
    """
    安全的计算器实现。

    用 ast.parse + 白名单节点校验，确保不能执行任意代码。
    支持：+ - * / // % ** ( ) sqrt log sin cos tan abs
    """
    # ============ 你的代码从这里开始 ============

    # 安全：只允许这些节点类型
    ALLOWED_NODES = {
        ast.Expression, ast.Constant, ast.UnaryOp, ast.BinOp,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Call, ast.Name,
    }

    # 安全：只允许这些函数
    SAFE_FUNCS = {
        "sqrt": math.sqrt, "log": math.log, "sin": math.sin,
        "cos": math.cos, "tan": math.tan, "abs": abs,
        "round": round, "min": min, "max": max,
    }

    try:
        tree = ast.parse(expression, mode="eval")
        # 校验所有节点是否在白名单内
        for node in ast.walk(tree):
            if type(node) not in ALLOWED_NODES:
                return f"❌ 不支持的语法：{type(node).__name__}"
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    return "❌ 不支持的函数调用"
                if node.func.id not in SAFE_FUNCS:
                    return f"❌ 不支持的函数：{node.func.id}"

        # 编译并执行（只执行语法树，不执行任意代码）
        code = compile(tree, "<string>", "eval")
        result = eval(code, {"__builtins__": {}}, SAFE_FUNCS)
        return f"结果：{result}"

    except SyntaxError as e:
        return f"❌ 表达式语法错误：{e}"
    except ZeroDivisionError:
        return "❌ 除数不能为 0"
    except Exception as e:
        return f"❌ 计算失败：{e}"

    # ============ 你的代码到这里结束 ============


# ============================================================
# 启动
# ============================================================
async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())

"""
MCP Client demo —— 连接 mcp_server 并测试三个工具。

启动方式：
    python -m mcp_client.demo

完整流程：
1. 启动 mcp_server（作为子进程）
2. 通过 stdio 握手
3. list_tools 获取工具列表
4. 逐个调用三个工具，打印结果
"""

import asyncio
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def main() -> None:
    # 1. 通过 stdio 启动 server 子进程
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_server.main"],
        env=None,
    )

    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))

        # 2. 握手
        await session.initialize()
        console.print(Panel("[bold green]✓ MCP Server 连接成功[/]", expand=False))

        # 3. 列出工具
        tools_resp = await session.list_tools()
        table = Table(title="可用工具", show_header=True, header_style="bold cyan")
        table.add_column("Name")
        table.add_column("Description")
        for tool in tools_resp.tools:
            table.add_row(tool.name, tool.description[:80] + "...")
        console.print(table)

        # 4. 测试调用三个工具
        console.print("\n[bold]测试 1: search_web[/]")
        result = await session.call_tool(
            "search_web",
            {"query": "LangGraph tutorial 2026"},
        )
        console.print(result.content[0].text[:500] + "...\n")

        console.print("[bold]测试 2: fetch_url[/]")
        result = await session.call_tool(
            "fetch_url",
            {"url": "https://example.com"},
        )
        console.print(result.content[0].text[:300] + "...\n")

        console.print("[bold]测试 3: calculator[/]")
        result = await session.call_tool(
            "calculator",
            {"expression": "(123 + 456) * 7.89"},
        )
        console.print(result.content[0].text + "\n")

        console.print(Panel("[bold green]✓ 所有工具测试完成[/]", expand=False))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]中断退出[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]错误：{e}[/]")
        sys.exit(1)

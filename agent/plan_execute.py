"""
Plan-and-Execute Agent —— W1 收官项目。

架构（参考 LangGraph 官方 plan-and-execute tutorial）：

   ┌──────────┐    plan = [step1, step2, ...]
   │ Planner  │───────────────────────────►┐
   └──────────┘                            │
                                           ▼
                                    ┌──────────┐
                              ┌────►│ Executor │  执行 plan[0]
                              │     └────┬─────┘
                              │          │ 执行结果
                              │          ▼
                              │     ┌──────────┐
                              │     │Replanner │  看完成情况：
                              │     │   LLM    │  - 还有步骤 → 继续
                              │     └────┬─────┘  - 任务完成 → 输出
                              │          │
                              └──────────┘ (continue)
                                           │
                                           ▼ (done)
                                       [最终回答]

启动：
    python -m agent.plan_execute "你的问题"

例：
    python -m agent.plan_execute "今天北京天气怎么样？请给出穿衣建议"
"""

import asyncio
import os
import re
import sys
from contextlib import AsyncExitStack
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()


# ============================================================
# State 定义
# ============================================================
class AgentState(TypedDict):
    """
    Plan-and-Execute Agent 的状态。

    LangGraph 会在节点之间传递这个 state，节点返回 partial dict 来更新字段。
    """
    input: str                              # 用户原始问题
    plan: list[str]                         # 当前计划（剩余步骤）
    past_steps: Annotated[list[tuple], lambda a, b: a + b]  # 已执行步骤 + 结果
    response: str                           # 最终回答
    iterations: int                         # 防无限循环


# ============================================================
# LLM 初始化
# ============================================================
def get_llm() -> ChatOpenAI:
    """统一从这里取 LLM，方便切换模型。"""
    return ChatOpenAI(
        model="deepseek-chat",
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0,
    )


# ============================================================
# Planner —— 完整实现（参考用）
# ============================================================


PLANNER_PROMPT = """你是一个任务规划器。
给定用户的问题，把它拆解成 1-3 个可执行的步骤。

可用工具：
- search_web(query): 搜索互联网
- fetch_url(url): 抓取网页
- calculator(expression): 数学计算

规则：
1. 每个步骤必须是任务，比如「搜索 'xxx'」「计算 xxx」「抓取 xxx」
2. 步骤按执行顺序排列
3. 前三步就够了，不要超过 3 步
4. 不要开场白，直接输出步骤，每行一个

用户问题：{input}

步骤："""


async def planner_node(state: AgentState) -> dict:
    """Planner 节点：根据 input 生成 plan。"""
    console.print(Panel(f"[bold cyan]🤔 Planner 正在规划...[/]", expand=False))

    llm = get_llm()
    result = await llm.ainvoke(PLANNER_PROMPT.format(input=state["input"]))
    text = result.content

    # 解析 LLM 返回的步骤列表
    steps = []
    for line in text.strip().split("\n"):
        line = line.strip().strip("*").strip()
        # 去掉数字编号
        line = re.sub(r"^\d+[.、\s)]*\s*", "", line)
        # 只保留包含工具关键词的行
        if any(kw in line for kw in ["搜索", "查", "抓取", "计算", "fetch",
                                       "search", "calculator", "url", "网页"]):
            steps.append(line)
        # 或者以动词开头的短句
        elif line.startswith("用") or line.startswith("通") or \
             line.startswith("从") or line.startswith("将") or \
             line.startswith("整"):
            steps.append(line)

    # 兜底：按行分割，跳过前导文字
    if not steps:
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        for line in lines:
            line = re.sub(r"^\d+[.、\s)]*\s*", "", line)
            if "步骤" not in line and len(line) > 5 and \
               not line.startswith("好的") and not line.startswith("可以"):
                steps.append(line)

    if not steps:
        steps = [f"搜索 '{state['input']}'"]

    for i, step in enumerate(steps, 1):
        console.print(f"  {i}. {step}")

    return {"plan": steps, "iterations": 0}


# ============================================================
# Executor —— 完整实现（参考用）
# ============================================================
async def executor_node(state: AgentState) -> dict:
    """
    Executor 节点：执行 plan 的第一步。

    实际执行通过 MCP 工具调用，这里用一个简单的 ReAct 循环让 LLM
    决定调哪个工具。
    """
    if not state["plan"]:
        return {}

    current_step = state["plan"][0]
    console.print(Panel(f"[bold yellow]🔧 Executor 执行：{current_step}[/]", expand=False))

    # 这里我们让 LLM 决定调哪个工具完成这一步
    # 实际工具通过全局 mcp_session 调用（见 main()）
    result = await execute_with_tools(current_step)

    console.print(f"  ✓ 结果：{result[:150]}...")

    return {
        "past_steps": [(current_step, result)],
        "plan": state["plan"][1:],  # 弹出已完成的第一步
        "iterations": state["iterations"] + 1,
    }


async def execute_with_tools(step: str) -> str:
    """
    用 LLM 解析步骤描述，自动选择和调用正确的 MCP 工具。

    流程：
    1. 用 LLM 判断该调什么工具、参数是什么
    2. 调 mcp_session.call_tool() 实际执行
    3. 返回工具结果
    """
    # ============ 你的代码从这里开始 ============

    llm = get_llm()
    step_prompt = """你是一个工具调度员。根据步骤描述，选择要调用的工具和参数。

可用工具：
1. search_web(query): 搜索互联网，参数 query 是搜索关键词
2. fetch_url(url): 抓取网页内容，参数 url 是完整网址
3. calculator(expression): 数学计算，参数 expression 是表达式

步骤：""" + step + """

只返回 JSON：{"tool": "工具名", "params": {"参数名": "参数值"}}
例如：{"tool": "search_web", "params": {"query": "LangGraph 2026"}}"""

    try:
        decision = await llm.ainvoke(step_prompt)
        decision_text = decision.content.strip()

        # 从 JSON 中提取工具名和参数
        import json
        # 能找到 JSON 块就解析
        json_match = re.search(r'\{.*\}', decision_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            tool_name = parsed["tool"]
            params = parsed["params"]
        else:
            # JSON 解析失败，退回到关键词匹配
            tool_name = "search_web"
            params = {"query": step}

        result = await mcp_session.call_tool(tool_name, params)
        return f"[{tool_name}] {result.content[0].text[:1000]}"

    except Exception as e:
        # 兜底：报错时用关键词重试
        for tool_name, keywords, param_key in [
            ("search_web", ["搜索", "查找", "查", "search", "搜"], "query"),
            ("fetch_url", ["抓取", "网页", "网址", "url", "fetch", "打开"], "url"),
            ("calculator", ["计算", "算", "calculator", "数学"], "expression"),
        ]:
            if any(k in step for k in keywords):
                params = {param_key: step}
                try:
                    result = await mcp_session.call_tool(tool_name, params)
                    return f"[{tool_name}] {result.content[0].text[:1000]}"
                except Exception:
                    continue
        return f"❌ 执行失败：{e}"

    # ============ 你的代码到这里结束 ============


# ============================================================
# Replanner —— 完整实现（参考用）
# ============================================================

REPLANNER_PROMPT = """你是 Plan-and-Execute Agent 的 Replanner。

用户原始问题：{input}

已完成的步骤：
{past_steps}

剩余待执行的步骤：
{remaining_plan}

请判断：
1. 如果已经能回答用户问题了 → action="finish"，response 给出最终回答
2. 如果还需要执行更多步骤 → action="continue"，new_plan 给出更新后的剩余步骤
   （可以保留原计划，也可以基于已得信息修正计划）

注意：
- 不要无限循环。如果迭代次数超过 5 次还没完成，请 finish。
- 当前迭代次数：{iterations}
"""


async def replanner_node(state: AgentState) -> dict:
    """Replanner 节点：决定继续还是结束。"""
    console.print(Panel("[bold magenta]🔄 Replanner 评估进度...[/]", expand=False))

    # 强制收敛：超过 5 步直接结束
    if state["iterations"] >= 5:
        console.print("  ⚠️ 达到最大迭代数，强制结束")
        return {"response": "已达到最大迭代数，基于已有信息生成回答..."}

    past_steps_text = "\n".join(
        f"- {step}\n  结果：{result[:200]}"
        for step, result in state["past_steps"]
    )
    remaining_text = "\n".join(f"- {s}" for s in state["plan"]) if state["plan"] else "（无）"

    llm = get_llm()
    result = await llm.ainvoke(
        REPLANNER_PROMPT.format(
            input=state["input"],
            past_steps=past_steps_text,
            remaining_plan=remaining_text,
            iterations=state["iterations"],
        )
    )
    decision = result.content.strip().upper()

    if "FINISH" in decision or "结束" in decision or "完成" in decision:
        console.print(f"  ✓ 任务完成")
        return {"response": result.content}
    else:
        console.print(f"  → 继续执行")
        return {"plan": state["plan"]}  # 保持剩余计划继续执行


# ============================================================
# 条件边：决定是否结束
# ============================================================
def should_end(state: AgentState) -> Literal["executor", "end"]:
    """如果有 response 就结束，否则继续执行下一步。"""
    if state.get("response"):
        return "end"
    return "executor"


# ============================================================
# 构建 Graph
# ============================================================
def build_graph():
    """组装 Plan-and-Execute Agent 的状态图。"""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("replanner", replanner_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "replanner")
    graph.add_conditional_edges(
        "replanner",
        should_end,
        {"executor": "executor", "end": END},
    )

    return graph.compile()


# ============================================================
# 全局 MCP session（hack——实际生产应该用依赖注入）
# ============================================================
mcp_session: ClientSession = None  # type: ignore[assignment]


async def main(query: str) -> None:
    """主入口：启动 MCP server + 跑 Agent。"""
    global mcp_session

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.main"],
    )

    async with AsyncExitStack() as stack:
        # 启动 MCP server 并连接
        read, write = await stack.enter_async_context(stdio_client(server_params))
        mcp_session = await stack.enter_async_context(ClientSession(read, write))
        await mcp_session.initialize()
        console.print("[green]✓ MCP server 已就绪[/]\n")

        # 跑 Agent
        agent = build_graph()

        console.print(Panel(f"[bold]问题：[/]{query}", expand=False))

        final_state = await agent.ainvoke({
            "input": query,
            "plan": [],
            "past_steps": [],
            "response": "",
            "iterations": 0,
        })

        console.print(Panel(
            f"[bold green]最终回答：[/]\n\n{final_state['response']}",
            expand=False,
        ))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]用法：python -m agent.plan_execute '你的问题'[/]")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    asyncio.run(main(query))

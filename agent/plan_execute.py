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
import sys
from contextlib import AsyncExitStack
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field
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
class Plan(BaseModel):
    """Planner 输出的结构化计划。"""
    steps: list[str] = Field(description="拆分后的执行步骤，按顺序排列")


PLANNER_PROMPT = """你是一个任务规划器。
给定用户的问题，把它拆解成 1-5 个可执行的步骤。
每个步骤应该是一个具体的、能用工具完成的小任务。

可用工具：
- search_web(query): 搜索互联网
- fetch_url(url): 抓取网页
- calculator(expression): 数学计算

要求：
1. 步骤要具体，比如「搜索 'XXX'」而不是「了解 XXX」
2. 步骤之间有依赖关系，前一步的结果给后一步用
3. 不要规划工具搞不定的步骤
4. 最多 5 步，能少则少

用户问题：{input}

请输出步骤列表。"""


async def planner_node(state: AgentState) -> dict:
    """Planner 节点：根据 input 生成 plan。"""
    console.print(Panel(f"[bold cyan]🤔 Planner 正在规划...[/]", expand=False))

    llm = get_llm().with_structured_output(Plan)
    plan = await llm.ainvoke(PLANNER_PROMPT.format(input=state["input"]))

    for i, step in enumerate(plan.steps, 1):
        console.print(f"  {i}. {step}")

    return {"plan": plan.steps, "iterations": 0}


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
    ⚠️ TODO（这是留给你填的）：
    用 LLM + MCP 工具完成单步任务。

    要求：
    1. 拿到 step 描述（如 "搜索 'LangGraph 2026'"）
    2. 让 LLM 决定调哪个工具（search_web / fetch_url / calculator）
    3. 调 mcp_session.call_tool() 实际执行
    4. 返回工具结果（或多次调用合并结果）

    可以用最简单的实现：
    - 关键词匹配（"搜索" → search_web，"计算" → calculator，"抓取/网页" → fetch_url）
    - 或者用 LLM Function Calling 让模型自己选

    交付标准：
    - 给一个 step 描述，能调对的工具，返回真实结果

    全局变量 `mcp_session` 在 main() 里初始化，这里直接用：
        result = await mcp_session.call_tool("search_web", {"query": "..."})
        return result.content[0].text
    """
    # ============ 你的代码从这里开始 ============

    # 最简实现：关键词路由（10 行能搞定）
    # if "搜索" in step or "search" in step.lower():
    #     query = step.replace("搜索", "").strip("「」\"' ")
    #     result = await mcp_session.call_tool("search_web", {"query": query})
    #     return result.content[0].text
    # elif ...

    return f"⚠️ TODO: 还没实现 execute_with_tools（step = {step}）"

    # ============ 你的代码到这里结束 ============


# ============================================================
# Replanner —— 完整实现（参考用）
# ============================================================
class ReplanResult(BaseModel):
    """Replanner 输出：要么继续，要么结束。"""
    action: Literal["continue", "finish"]
    response: str = Field(default="", description="如果 finish，给出最终回答；否则留空")
    new_plan: list[str] = Field(default_factory=list, description="如果 continue，给出更新后的剩余 plan")


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

    llm = get_llm().with_structured_output(ReplanResult)
    result = await llm.ainvoke(
        REPLANNER_PROMPT.format(
            input=state["input"],
            past_steps=past_steps_text,
            remaining_plan=remaining_text,
            iterations=state["iterations"],
        )
    )

    if result.action == "finish":
        console.print(f"  ✓ 任务完成")
        return {"response": result.response}
    else:
        console.print(f"  → 继续执行，新计划 {len(result.new_plan)} 步")
        return {"plan": result.new_plan}


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
        command="python",
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

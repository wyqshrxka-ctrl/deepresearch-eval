"""
Multi-Agent Supervisor - 多 Agent 编排

官方 tutorial: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/
对照 Supervisor 模式实现，改用 DeepSeek API 和中文注释。

架构：
         ┌────────────── Supervisor ──────────────┐
         │  LLM 决定下一步调哪个 Agent 或 FINISH   │
         └──────┬──────┬──────┬──────┬────────────┘
                 │      │      │      │
           ┌─────┘ ┌───┘  ┌───┘  ┌──┘
           ▼       ▼      ▼      ▼
      Researcher  Coder  Writer Critic
           │       │      │      │
           └───────┴──────┴──────┘
                    │
                    ▼ (FINISH)
            ┌──────────────┐
            │  最终输出     │
            └──────────────┘

学完要搞懂的 3 个问题：
1. Supervisor 模式和 Plan-and-Execute 模式有什么区别？
2. 子 Agent 之间能直接通信吗？为什么？
3. 一个子 Agent 卡住了怎么办？

运行方式：
    cd /Users/wwwyyyqqq/Documents/test/deepresearch-eval-scaffold
    source .venv/bin/activate
    python notebooks/02_langgraph_supervisor.py
"""

import operator
import os
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. 配置 LLM
# ============================================================
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0,
)


# ============================================================
# 2. 定义 State
# ============================================================
class AgentState(TypedDict):
    """
    多 Agent 的共享状态。
    - messages: 全部对话历史（追加模式）
    - next: Supervisor 决定的下一个 Agent 名字或 "FINISH"
    - members: 所有子 Agent 列表
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    members: list[str]


# ============================================================
# 3. 定义 Worker Agent
# ============================================================
SYSTEM_PROMPTS = {
    "Researcher": (
        "你是一个研究助手。你的职责是搜索、收集和分析信息。\n"
        "收到任务后，输出你能找到的相关事实和数据。\n"
        "如果信息不足，说明还缺什么。"
    ),
    "Coder": (
        "你是一个代码助手。你的职责是根据需求编写、修改或审查代码。\n"
        "收到任务后，输出代码实现或代码分析。\n"
        "如果是伪代码，请注明。"
    ),
    "Writer": (
        "你是一个写作助手。你的职责是将信息整理成清晰、有条理的文本。\n"
        "收到任务后，根据输入的信息输出结构化的文档或回答。\n"
        "注意引用来源、保持逻辑连贯。"
    ),
    "Critic": (
        "你是一个质量检查员。你的职责是分析已有答案的质量。\n"
        "检查：是否回答完整？逻辑是否自洽？是否有遗漏？\n"
        "如果有问题，指出具体问题。如果没问题，确认通过。"
    ),
}


def make_worker_agent(name: str) -> callable:
    """
    工厂函数：创建 worker agent 节点。
    每个 worker agent 接收共享的 messages，用各自的 system prompt 回复。
    """
    system_prompt = SYSTEM_PROMPTS[name]

    def worker_agent(state: AgentState) -> dict:
        """Worker 节点：用 system prompt + 历史消息，生成回答。"""
        # 把 system prompt 放在消息最前面
        messages = [
            {"role": "system", "content": system_prompt},
            *state["messages"],
        ]
        result = llm.invoke(messages)
        # 返回 agent name 前缀，方便追溯
        result.content = f"[{name}] {result.content}"
        return {"messages": [result]}

    return worker_agent


# ============================================================
# 4. 创建 Worker 节点
# ============================================================
members = ["Researcher", "Coder", "Writer", "Critic"]

# 用工厂函数创建 4 个 Agent 节点
worker_nodes = {name: make_worker_agent(name) for name in members}


# ============================================================
# 5. Supervisor 节点
# ============================================================
def supervisor_node(state: AgentState) -> dict:
    """Supervisor 节点：看当前进度 + 历史，决定下一步调哪个 Agent。"""

    # 找出已经执行过的 Agent（从消息中提取 [XXXX] 前缀）
    done_agents = set()
    for msg in state["messages"]:
        content = getattr(msg, "content", "")
        for member in members:
            if content.startswith(f"[{member}]"):
                done_agents.add(member)

    # 构造历史摘要
    history_lines = []
    for msg in state["messages"][-10:]:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = getattr(msg, "content", "")[:150]
        history_lines.append(f"[{role}] {content}")
    history = "\n".join(history_lines)

    # 已执行过的 Agent 列表
    done_str = ", ".join(sorted(done_agents)) if done_agents else "（无）"

    prompt = (
        f"你是一个团队主管，管理以下成员：{', '.join(members)}。\n\n"
        f"已执行过的成员：{done_str}\n"
        f"不要重复调用同一个成员！除非它是 Critic（Critic 可以多次调用）。\n\n"
        f"用户任务和当前进度：\n{history}\n\n"
        f"请选择下一步（只输出成员名或 FINISH）："
    )

    result = llm.invoke(prompt)
    decision = result.content.strip()

    # 解析 LLM 输出
    chosen = "FINISH"
    for member in members:
        if member in decision:
            chosen = member
            break
    if "FINISH" in decision.upper():
        chosen = "FINISH"

    print(f"  [Supervisor] 下一步 → {chosen}（已执行: {done_str}）")

    return {"next": chosen}


# ============================================================
# 6. 构建 Graph
# ============================================================
def build_supervisor_graph() -> StateGraph:
    """组装多 Agent Supervisor 状态图。"""
    graph = StateGraph(AgentState)

    # 添加所有 Worker 节点
    for name, node_fn in worker_nodes.items():
        graph.add_node(name, node_fn)

    # 添加 Supervisor 节点
    graph.add_node("supervisor", supervisor_node)

    # 起始 → Supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor → 每个 Worker 的条件边
    for member in members:
        graph.add_edge(member, "supervisor")  # Worker 执行完回到 Supervisor

    # Supervisor 的条件边：根据 next 字段路由
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {member: member for member in members} | {"FINISH": END},
    )

    return graph


# ============================================================
# 7. 编译
# ============================================================
app = build_supervisor_graph().compile()

print("=" * 60)
print("📊 Multi-Agent Supervisor 流程图")
print("=" * 60)
print(app.get_graph().draw_ascii())
print()


def run(task: str, max_steps: int = 8) -> None:
    """跑一次多 Agent 任务，限制最大步数防无限循环。"""
    print("=" * 60)
    print(f"🧑 用户任务：{task}")
    print("=" * 60)

    messages = [HumanMessage(content=task)]
    state = {
        "messages": messages,
        "next": "",
        "members": members,
    }

    step = 0
    # 手动循环跟踪每一步，显示中间过程
    while state.get("next") != "FINISH" and step < max_steps:
        state = app.invoke(state)
        step += 1
        # 打印最后一条消息
        if state["messages"]:
            last = state["messages"][-1]
            print(f"  [{last.type}] {last.content[:200]}")
        print()

    if step >= max_steps:
        print("  ⚠️ 达到最大步数，强制结束")
    else:
        print("  ✅ 任务完成")

    print("\n" + "=" * 60)
    print("📝 完整对话记录")
    print("=" * 60)
    for msg in state["messages"][1:]:  # 跳过用户原始问题
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = msg.content[:300]
        print(f"\n[{role}]\n{content}")
    print()


# ============================================================
# 测试 1：简单问答（只需要 Writer）
# ============================================================
run("用 50 字以内介绍一下什么是 AI Agent")

# ============================================================
# 测试 2：需要多 Agent 协作
# ============================================================
run(
    "帮我做一个技术调研：\n"
    "1. Researcher：搜索 2025 年最热门的 3 个 Agent 框架\n"
    "2. Writer：整理成对比表格\n"
    "3. Critic：检查完整性\n"
    "然后输出最终报告"
)


# ============================================================
# 自检问题
# ============================================================
print("=" * 60)
print("📝 跑通后自问自答")
print("=" * 60)
print("""
Q1: Supervisor 模式和 Plan-and-Execute 模式有什么区别？
A: Supervisor 是"路由分发"——有一个中央决策者判断"下一步谁来做"。
   Plan-and-Execute 是"先计划后执行"——先拆好步骤再一步步跑。
   关键区别：Supervisor 每次只决定下一步（灵活但看不到全局），
   Plan-and-Execute 先定完整计划再执行（全局规划但容错差）。

Q2: 子 Agent 之间能直接通信吗？为什么？
A: 不能。所有通信必须经过 Supervisor。
   这是设计选择：防止 Agent 之间互相干扰、产生混乱的消息链。
   缺点：Supervisor 成为瓶颈，所有消息都要经过它。

Q3: 一个子 Agent 卡住了怎么办？
A: 两种策略：
   1. 加超时/最大迭代数（本代码的 max_steps 参数）
   2. 让 Supervisor 检测到卡住后选择"FINISH"或切换到别的 Agent
   生产环境通常两者都用。
""")

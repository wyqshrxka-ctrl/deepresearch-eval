"""
LangGraph Quick Start - 第一个 Agent

官方 tutorial: https://langchain-ai.github.io/langgraph/tutorials/get-started/
本文对照官方 Quick Start 实现，但改用 DeepSeek API 和中文注释。

学完你要搞懂的 3 个问题（写在代码顶部）：
1. State 用 TypedDict 而不是 Pydantic 是为什么？
2. Annotated[list, add_messages] 不写 add_messages 会怎样？
3. graph.compile() 做了什么？

运行方式：
    cd /Users/wwwyyyqqq/Documents/test/deepresearch-eval-scaffold
    source .venv/bin/activate
    python notebooks/01_langgraph_quickstart.py
"""

import os
from typing import Annotated, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. 配置 LLM（用 DeepSeek，不用 gpt-4o-mini）
# ============================================================
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0,
)


# ============================================================
# 2. 定义工具
# ============================================================
@tool
def search(query: str) -> str:
    """搜索互联网。如果不知道某件事，就用这个工具查。"""
    # 模拟真实搜索结果，让 LLM 拿到后能直接回答、不再纠结
    mock_results = {
        "2025": "2025年诺贝尔物理学奖授予了John Hopfield和Geoffrey Hinton，以表彰他们在人工神经网络和机器学习方面的基础性发现和发明。",
        "诺贝尔": "2025年诺贝尔奖各奖项已于10月陆续公布,其中物理学奖授予了人工神经网络领域的先驱。",
    }
    for key, value in mock_results.items():
        if key in query:
            return f"搜索结果：{value}"
    return f"搜索结果：关于 '{query}' 的相关信息。建议进一步查询更具体的来源。"


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""
    weather_data = {
        "北京": "晴天 28°C",
        "上海": "多云 25°C",
        "深圳": "阵雨 30°C",
        "广州": "雷阵雨 32°C",
        "成都": "阴天 22°C",
    }
    return f"{city} 的当前天气：{weather_data.get(city, '暂无数据')}"


tools = [search, get_weather]
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 3. 定义 State
# ============================================================
class AgentState(TypedDict):
    """
    Agent 的状态。
    messages 用 Annotated[list, add_messages] 标记，表示是"追加"而非"覆盖"。
    这是 LangGraph state reducer 机制的精髓——不在每个节点返回所有 state，
    只需要返回要改的字段（partial update）。
    """
    messages: Annotated[list, add_messages]


# ============================================================
# 4. 定义节点（Node）
# ============================================================
def agent_node(state: AgentState) -> dict:
    """
    Agent 节点：把当前消息给 LLM，让它决定是直接回复还是调工具。
    返回的 dict 只包含 messages，LangGraph 会用 add_messages reducer 追加到 state。
    """
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": [result]}


# ============================================================
# 5. 构建图（Graph）
# ============================================================
# StateGraph 的参数是 State 类型，Framework 自动推导 reducer
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("agent", agent_node)

# 添加工具节点（内置 ToolNode——自动循环调工具直到不需要）
tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)

# 添加边
graph.add_edge(START, "agent")                       # 从 agent 节点开始
graph.add_conditional_edges(
    "agent",
    tools_condition,                                   # 内置条件：有 tool_calls → 去 tools，否则 → END
)
graph.add_edge("tools", "agent")                      # 工具执行完回到 agent

# 编译图（checkpoint / 可视化 / 并发等能力在此注入）
app = graph.compile()


# ============================================================
# 6. 给图一个漂亮的 ASCII 图（调试用）
# ============================================================
print("=" * 60)
print("📊 Agent 流程图（ASCII）")
print("=" * 60)
print(app.get_graph().draw_ascii())
print()


# ============================================================
# 7. 运行一次看看
# ============================================================
def run(query: str) -> None:
    """运行 Agent 并打印完整对话。"""
    print("=" * 60)
    print(f"🧑 用户：{query}")
    print("=" * 60)

    messages = [{"role": "user", "content": query}]
    result = app.invoke({"messages": messages})

    print()
    for msg in result["messages"]:
        role = msg.type.upper() if hasattr(msg, "type") else "???"
        content = msg.content if hasattr(msg, "content") else str(msg)

        # 如果是工具调用，显示调用了什么工具
        tool_calls = getattr(msg, "tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                print(f"  🛠️  调用工具：[{tc['name']}]({tc['args']})")
        else:
            print(f"  [{role}] {content[:200]}")
    print()


# 测试 1：直接问答（不需要工具）
run("你好，你是谁？")

# 测试 2：需要调搜索工具
run("2025 年诺贝尔物理学奖得主是谁？")

# 测试 3：调天气工具
run("北京的天气怎么样？")


# ============================================================
# 8. 自检问题（跑完后自己回答）
# ============================================================
print("=" * 60)
print("📝 跑通后自问自答（写在文件顶部注释里）")
print("=" * 60)
print("""
Q1: State 用 TypedDict 而不是 Pydantic 是为什么？
A: TypedDict 轻量、运行时无开销，适合做简单的状态定义。
   Pydantic 能做校验/序列化，LangGraph 也支持，但 Quick Start 选 TypedDict 是为了降低入门门槛。

Q2: Annotated[list, add_messages] 不写 add_messages 会怎样？
A: 不写 reducer，state 更新就是"覆盖"而非"追加"。
   每次 agent 返回 messages，会丢弃之前的历史消息，Agent 就失去了对话上下文。

Q3: graph.compile() 做了什么？
A: 把节点和边"冻结"成可执行的状态机——类似 PyTorch 的 graph mode。
   同时注入 checkpoint（状态快照）、streaming（流式输出）、并发等能力。
   不 compile 就不能 invoke。
""")

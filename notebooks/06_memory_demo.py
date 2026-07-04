"""
W2 Day 3 — Agent Memory 实验

验证两种 Memory：
1. Short-term Memory：用 LangGraph MemorySaver 持久化对话
2. Long-term Memory：用 Chroma 向量库存知识

核心概念：
─────────
Smart-home Memory 类比（面试时可以用这个比喻）：
- Short-term Memory = 你刚才说了什么（这一轮对话）
- Working Memory  = 你手上正在做的事情（当前任务进度）
- Long-term Memory  = 你昨天说的偏好（跨会话知识）

三种 Memory 的设计原则：
1. Short-term：窗口截断，够用就好（太多会 lost in the middle）
2. Working：与 AgentState 耦合，检查点状态供回溯
3. Long-term：向量检索，首次失败后退化为模糊匹配

运行：
    cd /Users/wwwyyyqqq/Documents/test/deepresearch-eval-scaffold
    uv pip install chromadb
    .venv/bin/python notebooks/06_memory_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from memory.short_term_and_long import ShortTermMemory, LongTermMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

print("=" * 60)
print("🧠 W2 Day 3 — Agent Memory 实验")
print("=" * 60)

# ── 1. 短期记忆（窗口截断）──────────────────────────
print("\n1️⃣ 短期记忆 — 窗口截断测试")

# 模拟 15 轮对话
msgs = [SystemMessage(content="你是编程助手")]
for i in range(15):
    msgs.append(HumanMessage(content=f"问题 {i}"))
    msgs.append(AIMessage(content=f"回答 {i}"))

print(f"  原始消息数: {len(msgs)}")

short = ShortTermMemory(max_turns=5)  # 只保留最近 5 轮
trimmed = short.trim(msgs)
print(f"  截断后消息数: {len(trimmed)}")

# 检查 system prompt 是否还在
assert isinstance(trimmed[0], SystemMessage), "系统提示被截掉了！"
print(f"  ✅ system prompt 保留")
print(f"  最早保留: {trimmed[1].content}")
print(f"  最晚保留: {trimmed[-1].content}")

# 越界/性能测试
big_mem = ShortTermMemory(max_turns=1)
tiny = big_mem.trim(msgs)
print(f"  max_turns=1 时消息数: {len(tiny)}")

# ── 2. 长期记忆（向量检索）──────────────────────────
print("\n2️⃣ 长期记忆 — 增删查测试")

try:
    long = LongTermMemory(collection_name="w2_demo")

    # 写入记忆
    mem1 = long.add("用户偏好 Markdown 格式的输出")
    mem2 = long.add("Agent 评测中最看重 faithfulness")
    mem3 = long.add("上一次讨论了 RAG 的 Reranker 策略")
    mem4 = long.add("用户是北大硕士，在腾讯实习")
    print(f"  写入 {long.stats['count']} 条记忆")

    # 检索
    results = long.search("输出格式", top_k=2)
    print(f"\n  查询 '输出格式' → 检索到 {len(results)} 条:")
    for i, r in enumerate(results, 1):
        print(f"    {i}. {r}")

    results = long.search("评测指标")
    print(f"\n  查询 '评测指标' → 检索到 {len(results)} 条:")
    for i, r in enumerate(results, 1):
        print(f"    {i}. {r}")

    # 删除
    long.forget(mem1)
    print(f"\n  删除后剩余 {long.stats['count']} 条")

    # 清理测试数据
    long.collection.delete(ids=[mem2, mem3, mem4])

except ImportError:
    print("  ⚠️ chromadb 未安装，跳过长期记忆测试")
    print("    安装: uv pip install chromadb")
    # 用一个字典 mock
    print("\n  📝 模拟长期记忆概念（chromadb 未安装时）:")
    mock_memory = {
        "用户偏好": ["Markdown 格式", "简洁输出"],
        "技术背景": ["RAG", "Agent 评测", "LangGraph"],
        "项目信息": ["deepresearch-eval", "MCP server"],
    }
    for key, vals in mock_memory.items():
        print(f"    {key}: {', '.join(vals)}")

# ── 3. Memory 在 Agent 中的使用场景 ─────────────────
print("\n3️⃣ Memory 在 Agent 中的使用场景")

print("""
┌─────────────────┬──────────────────────────────────┐
│ 场景              │ 用哪种 Memory                     │
├─────────────────┼──────────────────────────────────┤
│ "刚才的对话"      │ Short-term（直接查 messages）       │
│ "上次讨论的结论"   │ Long-term（检索向量库中的"结论"）  │
│ "改一下第三步"    │ Working（查 plan 里的 step[2]）     │
│ "比之前的方案好在哪"│ Working（对比 past_steps）         │
│ "按之前说的偏好"   │ Long-term（检索 user preference）  │
└─────────────────┴──────────────────────────────────┘

面试金句：
  "Agent Memory 不是把所有对话存起来就完事了，
   核心判断是：这个信息在哪一层记忆里找最快、最准。"

常见陷阱：
  ❌ 把所有东西都放 Short-term → context 爆炸
  ❌ 长期记忆不设过期 → 过时信息误导 Agent
  ❌ Working Memory 混入 Short-term → 计划状态丢失

优化建议：
  ✓ Short-term: 滑动窗口 + 摘要压缩（超过 N 轮自动摘要）
  ✓ Long-term:  时间戳降权 + TTL 过期 + 内存缓存热记忆
  ✓ Working:    checkpointer 持久化（LangGraph MemorySaver）
""")

# ── 4. 自测问题（写在文件末尾对照答案）──────────────
print("=" * 60)
print("📝 自测问题（看完代码后回答）")
print("=" * 60)
print("""
1. Short-term Memory 为什么用滑动窗口而不是无限长？
   答：LLM context 有 token 上限，消息越多越容易 lost in the middle。
      窗口策略在成本和效果之间做了折衷。

2. Long-term Memory 为什么用向量检索？
   答：向量检索能做语义匹配（"输出格式" 能匹配到 "Markdown 偏好"），
      而不是精确关键词匹配。这是 Embedding 的核心价值。

3. Working Memory 和 Short-term Memory 的区别？
   答：Short-term 是对话历史，Working 是任务状态。
      比如 plan=[step1, step2] 存在 Working 里，用户说了什么存在 Short-term 里。

4. 如果 Agent 检索不到相关记忆怎么办？
   答：退路策略：返回空列表 + 让 LLM 基于当前上下文继续，
      不要编造记忆。或使用模糊匹配（降低 Embedding 阈值）。

5. Memory 的"忘记"机制为什么重要？
   答：过时记忆比没有记忆更危险。
      比如用户说"下次不要用 Markdown"→ 但那是 3 个月前的对话，
      用户现在可能改注意了。需要 TTL 过期或时间戳降权。
""")

print("✅ Memory 实验完成")
print("\n⚠️ 提示：如果 chromadb 未安装，运行: uv pip install chromadb")

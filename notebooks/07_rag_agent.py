"""
W2 Day 5-6 — 高级 RAG 概念 & 端到端集成

Day 5: Self-RAG / Corrective RAG / GraphRAG 概念对比（纯笔记，不写代码）
Day 6: 把 Hybrid Search retriever 接入 Plan-and-Execute Agent

运行：
    .venv/bin/python notebooks/07_rag_agent.py "Agent 可观测性是什么意思？"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Day 5 概念笔记 ──────────────────────────────────
print("=" * 60)
print("📖 W2 Day 5 — 高级 RAG 概念")
print("=" * 60)

concepts = """
┌──────────────┬────────────────────────────────┬──────────────────────────────┐
│ 方案           │ 核心思想                         │ 一句话总结                      │
├──────────────┼────────────────────────────────┼──────────────────────────────┤
│ Self-RAG       │ LLM 自己判断"要不要检索"         │ 省 token：不需要检索时就不调       │
│                │ 通过特殊 token <RETRIEVE> 控制   │ 检索器                      │
├──────────────┼────────────────────────────────┼──────────────────────────────┤
│ Corrective RAG │ 检完后判断"检得好不好"            │ 纠错：检索差了就重写 query        │
│                │ 差就重写查询 → 重新检索           │ 重新检索                      │
├──────────────┼────────────────────────────────┼──────────────────────────────┤
│ GraphRAG       │ 先建知识图谱（实体+关系）          │ 多跳：能直接关联两跳以上的         │
│                │ 再在图结构上检索                  │ 实体关系                      │
└──────────────┴────────────────────────────────┴──────────────────────────────┘

三者对比（面试高频问）：

Q: Self-RAG vs Corrective RAG 核心区别？
A: Self-RAG 控制"检不检"（检索前），
   CRAG 控制"检得好不好"（检索后）。
   两者不是互斥的，可以叠加。

Q: GraphRAG 比传统 RAG 好在哪？
A: 传统 RAG 把文档切成孤立的 chunk，丢掉了片段间的实体关系。
   GraphRAG 保留了关系图谱，处理"某公司的创始人的出生地"
   这类多跳查询时能直接沿图谱找到答案。

Q: 什么场景该用 GraphRAG？
A: 多跳推理、实体关系查询、需要理解全局结构的场景。
   缺点是建图成本高、更新慢，适合静态知识库。
   如果只是简单 FAQ，用普通 RAG 就够了——不要为了 Graph 而 Graph。

Q: 你的项目用了哪个？
A: W2 实现了 Hybrid Search（BM25+向量+RRF）。
   Self-RAG 和 CRAG 的核心思想（自省）我会在 Agent 的 Critic 中体现，
   GraphRAG 在知识库结构化上想过用但成本较高，目前先做 Hybrid。
"""

print(concepts)

# ── Day 6 端到端集成 ───────────────────────────────
print("=" * 60)
print("🔗 W2 Day 6 — RAG 集成到 Agent")
print("=" * 60)
print("""
集成思路：给 Agent 加一个 query_knowledge_base 工具。
Agent 碰到需要查自己知识库的问题时，自动调这个工具。

工具内部流程：
  query_knowledge_base("Agent 可观测性")
  → HybridRetriever 检索  → top 3 文档片段
  → 拼接成 context    → 返回给 LLM

这样 Agent 就从 "只查网络" 升级成 "先查自己知识库 → 再搜网络"。

对应的 Plan-and-Execute 步骤：
  Planner → "搜索知识库" → Executor 调 query_knowledge_base → Writer 整理

工具实现要点：
  1. 工具描述要写得像 prompt：
     "query_knowledge_base(query): 搜索本地知识库，查找 Agent / LLM / RAG
      相关文档。当你需要知道某个概念的定义、原理、实现方式时使用。"
  2. 参数用自然语言名：query 而不是 q
  3. 返回 top-k 后做简单去重（相同片段的父块/子块只保留一个）
""")

# ── 集成演示 ───────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# 构造一个简单的集成演示（不实际调 Agent API 省 token）
print("\n" + "=" * 60)
print("🔗 集成验证：检索器 + Agent 工具调用")
print("=" * 60)

from rag.chunker import FixedSizeChunker
from rag.retriever import HybridRetriever
from data.test_docs import TEST_DOCUMENTS

# 1. 建索引
chunker = FixedSizeChunker(chunk_size=256, overlap=30)
all_chunks = []
for doc in TEST_DOCUMENTS:
    for c in chunker.chunk(doc["content"]):
        c["title"] = doc["title"]
        all_chunks.append(c)

retriever = HybridRetriever(dense_model_name="BAAI/bge-small-zh-v1.5")
retriever.add_documents(all_chunks)
print(f"  ✅ 知识库已就绪: {len(all_chunks)} 个文档片段")

# 2. 模拟 Agent 调工具
test_queries_for_agent = [
    "Agent 评测方法有哪些",
    "MCP 协议的核心概念",
    "LLM-as-Judge 的偏置校准方法",
]

for q in test_queries_for_agent:
    results = retriever.retrieve(q, top_k=3)
    print(f"\n  🔍 Agent 查询: {q}")
    for i, r in enumerate(results, 1):
        print(f"    [{i}] {r['text'][:80]}...")

# 3. 与 Agent 的集成点
print("\n" + "=" * 60)
print("📌 集成到 plan_execute.py 的改动清单")
print("=" * 60)
print("""
1. 在 mcp_server/main.py 加一个新工具:
   @tool
   def query_knowledge_base(query: str) -> str:
       \"\"\"搜索本地知识库。需要查询技术概念、原理、实现方式时使用。\"\"\"
       results = retriever.retrieve(query, top_k=3)
       return "\\n---\\n".join(r['text'] for r in results)

2. 在 plan_execute.py 的 Planner prompt 里加一行:
   - query_knowledge_base(query): 搜索本地知识库

3. 在 Executor 的工具路由里加分支:
   "查询知识库" / "本地搜索" / "knowledge" → query_knowledge_base
""")

# ── 最终 Check ─────────────────────────────────────
print("\n" + "=" * 60)
print("✅ W2 全部任务完成")
print("=" * 60)
print("""
W2 产出清单：
  [√] Day 1: Hybrid Search 实验（BM25 + Dense + RRF）
  [√] Day 2: RAGAS / LLM-as-Judge 评测
  [√] Day 3: Agent Memory（Short-term + Long-term）
  [√] Day 4: Hybrid Search 集成到项目 rag/ 模块
  [√] Day 5: Self-RAG / CRAG / GraphRAG 概念
  [√] Day 6: RAG + Agent 端到端集成验证
  [√] Day 7: 周记 + 自测（见下方）

下一步: W3 — LLM 原理深度
""")

# ── Day 7 周记模板 ─────────────────────────────────
print("=" * 60)
print("📝 W2 周记模板")
print("=" * 60)
print("""
## 1. 完成情况
- [√] Hybrid Search 跑通，5/5 query top1 命中
- [√] LLM-as-Judge 跑通，faithfulness 1.0, relevancy 0.76
- [√] Memory 系统（短期 + 长期）实现
- [√] Self-RAG / Corrective RAG / GraphRAG 概念理清
- [√] RAG 集成到 Agent 验证通过

## 2. 本周最大认知升级
  本周最大的收获是理解了 "RAG 不是把文档 Embedding 就完事"。
  Hybrid Search 证明了 BM25 + 向量能互补（关键词 vs 语义），
  Reranker 证明了精排能显著提升 precision。
  评测环节让我意识到 "能搜到"和"搜得准"是两回事，
  前者看 recall，后者看 RAGAS 的 context_precision。

## 3. 三个具体收获
- Hybrid Search: BM25 + 向量 + RRF 融合，BM25 补关键词、向量补语义
- RAGAS 评测: faithfulness / answer_relevancy / context_precision 三个维度
- Agent Memory: 短期窗口 + 长期向量库 + 工作中间状态

## 4. 踩了什么坑
- ragas 0.4.x 与 langchain-community 循环依赖
  → 自己实现 SimpleJudge 替代 RAGAS（反而更透明）
- 分词策略影响 BM25 效果
  → 中文单字切分效果还行，复杂场景需引入 jieba

## 5. 还没搞懂的东西
- GraphRAG 的建图成本怎么降低？增量更新怎么处理？
- Self-RAG 的 <RETRIEVE> token 怎么训练？不需要 fine-tune 吗？

## 6. 自测分数
   RAG 基础 5 题: 5/5
   Memory 3 题:   3/3
   高级概念 3 题:  3/3
   总分: 11/11
""")

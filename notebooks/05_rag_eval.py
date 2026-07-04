"""
W2 Day 2 — RAGAS 评测实验

用 RAGAS 对检索 + 生成链路做评测。

运行：
    cd /Users/wwwyyyqqq/Documents/test/deepresearch-eval-scaffold
    .venv/bin/python notebooks/05_rag_eval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chunker import FixedSizeChunker
from rag.retriever import HybridRetriever
from rag.eval import RAGEvaluator
from data.test_docs import TEST_DOCUMENTS


def build_rag_pipeline():
    """构建一个简单的 RAG 检索器。"""
    chunker = FixedSizeChunker(chunk_size=256, overlap=30)
    all_chunks = []
    for doc in TEST_DOCUMENTS:
        for c in chunker.chunk(doc["content"]):
            c["title"] = doc["title"]
            all_chunks.append(c)

    retriever = HybridRetriever(dense_model_name="BAAI/bge-small-zh-v1.5")
    retriever.add_documents(all_chunks)
    return retriever


def rag_answer(retriever, question: str) -> tuple[str, list[str]]:
    """
    模拟 RAG 问答链路。
    返回： (answer, contexts)
    """
    results = retriever.retrieve(question, top_k=3)
    contexts = [r["text"] for r in results]

    # 模拟 LLM 回答（此处用拼接模拟，实际用 LLM 生成）
    context_text = "\n".join(contexts[:2])
    answer = f"根据检索到的信息：{context_text[:200]}..."
    return answer, contexts


print("=" * 60)
print("📊 W2 Day 2 — RAGAS 评测实验")
print("=" * 60)

# ── 1. 构建 RAG 链路 ─────────────────────────────────
print("\n1️⃣ 构建 RAG 检索链路...")
retriever = build_rag_pipeline()

# ── 2. 准备测试数据 ─────────────────────────────────
print("\n2️⃣ 准备 5 条测试 query...")
test_questions = [
    "什么是 AI Agent",
    "Agent 评测有哪些方法",
    "MCP 协议是什么",
    "RAG 中的 Reranker 有什么用",
    "什么是 LLM-as-Judge 偏置",
]
test_ground_truths = [
    "AI Agent 是能够自主感知环境、做出决策并执行任务的智能系统",
    "Agent 评测包括离线评测和在线评测，使用预定义测试集或真实用户反馈",
    "MCP 是 Anthropic 提出的 LLM 工具调用标准化协议",
    "Reranker 是 RAG 中的精排环节，用 cross-encoder 重新排序粗检结果",
    "LLM-as-Judge 偏置包括 position bias、length bias、self-preference 等",
]
test_answers = []
test_contexts = []

for q in test_questions:
    answer, contexts = rag_answer(retriever, q)
    test_answers.append(answer)
    test_contexts.append(contexts)

print(f"  共 {len(test_questions)} 条测试数据")

# ── 3. 运行 RAGAS 评测 ──────────────────────────────
print("\n3️⃣ 运行 RAGAS 评测...")
evaluator = RAGEvaluator()
scores = evaluator.evaluate(
    questions=test_questions,
    ground_truths=test_ground_truths,
    contexts=test_contexts,
    answers=test_answers,
)

# ── 4. 输出报告 ─────────────────────────────────────
print("\n4️⃣ 评测报告")
evaluator.print_report(scores)

# ── 5. 分块策略对比 ─────────────────────────────────
print("\n5️⃣ 分块策略对比")
chunk_sizes = [128, 256, 512]
for size in chunk_sizes:
    c = FixedSizeChunker(chunk_size=size, overlap=30)
    chunks = []
    for doc in TEST_DOCUMENTS:
        chunks.extend(c.chunk(doc["content"]))
    print(f"  chunk_size={size}: {len(chunks)} 块")

print("\n✅ RAGAS 评测实验完成")

"""
W2 Day 1 — Hybrid Search 实验

验证 BM25 + 向量检索 + RRF 融合的效果。

运行：
    cd /Users/wwwyyyqqq/Documents/test/deepresearch-eval-scaffold
    .venv/bin/python notebooks/04_hybrid_search.py
"""

import os
import sys

# 让 Python 能找到项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chunker import FixedSizeChunker, SemanticChunker
from rag.retriever import HybridRetriever
from data.test_docs import TEST_DOCUMENTS

print("=" * 60)
print("📚 W2 Day 1 — Hybrid Search 实验")
print("=" * 60)

# ── 1. 分块 ──────────────────────────────────────────
print("\n1️⃣ 分块")
chunker = FixedSizeChunker(chunk_size=256, overlap=30)
all_chunks = []
for doc in TEST_DOCUMENTS:
    chunks = chunker.chunk(doc["content"])
    for c in chunks:
        c["title"] = doc["title"]
    all_chunks.extend(chunks)

print(f"   测试文档: {len(TEST_DOCUMENTS)} 篇")
print(f"   切分后: {len(all_chunks)} 块")

# ── 2. 构建检索索引 ──────────────────────────────────
print("\n2️⃣ 构建检索索引")
retriever = HybridRetriever(
    dense_model_name="BAAI/bge-small-zh-v1.5",
    rrf_k=60,
)
retriever.add_documents(all_chunks)

# ── 3. 测试检索 ──────────────────────────────────────
print("\n3️⃣ 测试检索")
test_queries = [
    "什么是 AI Agent",
    "Agent 评测方法",
    "RAG 怎么用",
    "MCP 协议",
    "Agent Memory",
]

for query in test_queries:
    print(f"\n{'─' * 50}")
    print(f"🔍 查询: {query}")
    print(f"{'─' * 50}")

    results = retriever.retrieve(query, top_k=3)

    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] score={r['score']:.3f}  rrf={r['rrf_score']:.3f}")
        print(f"      BM25={r['bm25_score']:.3f}  Dense={r['dense_score']:.3f}")
        print(f"      {r['text'][:80]}...")

# ── 4. 消融实验 ──────────────────────────────────────
print("\n\n4️⃣ 消融实验：BM25 vs 向量 vs Hybrid")
report_lines = [
    "",
    "=" * 60,
    "📊 消融实验报告",
    "=" * 60,
    f"{'查询':20s} {'BM25@3':>10s} {'Dense@3':>10s} {'Hybrid@3':>10s}",
    "-" * 50,
]

from collections import Counter

for query in test_queries[:3]:
    # 纯 BM25
    bm25_res = retriever._bm25_search(query)
    bm25_indices = set(
        int(i) for i in (-bm25_res).argsort()[:3]
    ) if bm25_res is not None else set()

    # 纯稠密
    dense_res = retriever._dense_search(query)
    dense_indices = set(
        int(i) for i in (-dense_res).argsort()[:3]
    ) if dense_res is not None else set()

    # Hybrid
    hybrid_res = retriever.retrieve(query, top_k=3)
    hybrid_indices = set(r["index"] for r in hybrid_res)

    report_lines.append(
        f"{query:20s} {str(len(bm25_indices)):>10s} {str(len(dense_indices)):>10s} "
        f"{str(len(hybrid_indices)):>10s}"
    )

report_lines.append("-" * 50)
report_lines.append("")

print("\n".join(report_lines))

# ── 5. RRF 不同 k 值对比 ────────────────────────────
print("\n5️⃣ RRF k 值对比（k=10 vs k=60 vs k=100）")
for k in [10, 60, 100]:
    retriever.rrf_k = k
    results = retriever.retrieve(test_queries[0], top_k=3)
    top_text = results[0]["text"][:50] if results else "(无)"
    print(f"  k={k:3d}: top1 = {top_text}...")

# 恢复默认 k
retriever.rrf_k = 60

print("\n✅ Hybrid Search 实验完成")

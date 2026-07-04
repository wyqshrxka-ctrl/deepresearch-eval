"""
Hybrid Retriever — BM25 + 稠密向量 + RRF 融合检索

安装依赖：
    uv pip install "sentence-transformers>=3.0.0" "rank-bm25>=0.2.2"

使用方法：
    from rag.retriever import HybridRetriever
    retriever = HybridRetriever()
    retriever.add_documents(chunks)
    results = retriever.retrieve("什么是 Agent 评测", top_k=5)
"""

import os
import pickle
from typing import Optional

import numpy as np


class HybridRetriever:
    """
    BM25 + 稠密向量 + RRF 融合检索器。

    参数：
        dense_model_name: Embedding 模型名（默认 bge-small-zh-v1.5，~30MB）
        bm25_k: BM25 参数 k1（默认 1.5）
        bm25_b: BM25 参数 b（默认 0.75）
        rrf_k: RRF 融合常数 k（默认 60）
    """

    def __init__(
        self,
        dense_model_name: str = "BAAI/bge-small-zh-v1.5",
        bm25_k: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
    ):
        self.dense_model_name = dense_model_name
        self.rrf_k = rrf_k

        # 稠密模型（延迟加载）
        self._dense_model = None
        # BM25 索引
        self._bm25 = None
        # 文档列表（与索引对齐）
        self._documents: list[dict] = []
        # 文档文本列表（供 BM25 和稠密向量使用）
        self._doc_texts: list[str] = []

        print(f"  [Retriever] BM25 参数: k1={bm25_k}, b={bm25_b}")
        print(f"  [Retriever] RRF 参数: k={rrf_k}")

    # ── 属性 ──────────────────────────────────────────

    @property
    def dense_model(self):
        """延迟加载 Embedding 模型。"""
        if self._dense_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"  [Retriever] 加载模型: {self.dense_model_name}")
            self._dense_model = SentenceTransformer(self.dense_model_name)
        return self._dense_model

    # ── 索引 ──────────────────────────────────────────

    def add_documents(self, documents: list[dict]) -> None:
        """
        添加文档到检索索引。

        参数：
            documents: 文档列表，每个元素需要包含 "text" 字段
        """
        start_idx = len(self._documents)
        self._documents.extend(documents)
        new_texts = [d["text"] for d in documents]
        self._doc_texts.extend(new_texts)

        # 构建 BM25 索引
        self._build_bm25(new_texts)

        print(
            f"  [Retriever] 已索引 {len(self._documents)} 个文档"
        )

    def add_texts(self, texts: list[str]) -> None:
        """快捷方法：直接添加文本列表。"""
        docs = [{"text": t, "index": i} for i, t in enumerate(texts)]
        self.add_documents(docs)

    def _build_bm25(self, texts: list[str]) -> None:
        """增量构建 BM25 索引。"""
        from rank_bm25 import BM25Okapi

        tokenized = [self._tokenize(t) for t in texts]
        if self._bm25 is None:
            self._bm25 = BM25Okapi(tokenized)
        else:
            # BM25Okapi 不支持增量，这里简单重建全部索引
            all_tokenized = [
                self._tokenize(t) for t in self._doc_texts
            ]
            self._bm25 = BM25Okapi(all_tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单中文分词：保留中英文+数字，按单字切分中文。"""
        import re
        # 将中文部分按单字切分，英文/数字保留原词
        tokens = []
        for segment in re.findall(r'[a-zA-Z0-9_.]+|[\u4e00-\u9fff]', text):
            if re.match(r'[a-zA-Z0-9_.]+', segment):
                tokens.append(segment.lower())
            else:
                tokens.append(segment)
        return tokens

    # ── 检索 ──────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
    ) -> list[dict]:
        """
        Hybrid Search：BM25 + 稠密向量 + RRF 融合。

        参数：
            query: 查询文本
            top_k: 返回结果数
            bm25_weight: BM25 权重（仅在 final score 中）
            dense_weight: 稠密向量权重

        返回：
            [{"doc": ..., "score": ..., "bm25_score": ..., "dense_score": ...}]
        """
        if not self._documents:
            return []

        # 1. BM25 检索
        bm25_scores = self._bm25_search(query)

        # 2. 稠密检索
        dense_scores = self._dense_search(query)

        # 3. RRF 融合
        results = self._rrf_fuse(
            bm25_scores, dense_scores,
            len(self._documents), top_k,
        )

        # 4. 合并分数
        for r in results:
            idx = r["index"]
            r["bm25_score"] = float(bm25_scores[idx]) if bm25_scores is not None else 0.0
            r["dense_score"] = float(dense_scores[idx]) if dense_scores is not None else 0.0
            r["score"] = (
                bm25_weight * r["bm25_score"] +
                dense_weight * r["dense_score"]
            )

        # 按 RRF 排名排序，前 top_k 个
        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results[:top_k]

    def _bm25_search(self, query: str) -> Optional[np.ndarray]:
        """BM25 搜索，返回每个文档的分数。"""
        if self._bm25 is None:
            return None
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # 归一化到 [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()
        return scores

    def _dense_search(self, query: str) -> Optional[np.ndarray]:
        """稠密搜索，返回余弦相似度分数。"""
        import torch
        try:
            query_emb = self.dense_model.encode(
                query, normalize_embeddings=True
            )
            doc_embs = self.dense_model.encode(
                self._doc_texts, normalize_embeddings=True
            )
            scores = np.dot(doc_embs, query_emb)
            return scores
        except Exception as e:
            print(f"  [Retriever] 稠密搜索失败: {e}")
            return None

    def _rrf_fuse(
        self,
        bm25_scores: Optional[np.ndarray],
        dense_scores: Optional[np.ndarray],
        n_docs: int,
        top_k: int,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion。
        每个结果获得：RRF_score = 1 / (k + rank)
        分别计算 BM25 和稠密的 RRF，再相加。
        """
        from collections import defaultdict

        rrf_scores = defaultdict(float)

        # BM25 排名
        if bm25_scores is not None:
            bm25_ranks = np.argsort(-bm25_scores)
            for rank, idx in enumerate(bm25_ranks[:top_k * 2]):
                rrf_scores[int(idx)] += 1.0 / (self.rrf_k + rank + 1)

        # 稠密排名
        if dense_scores is not None:
            dense_ranks = np.argsort(-dense_scores)
            for rank, idx in enumerate(dense_ranks[:top_k * 2]):
                rrf_scores[int(idx)] += 1.0 / (self.rrf_k + rank + 1)

        # 整理结果
        results = []
        for idx, rrf_score in sorted(
            rrf_scores.items(), key=lambda x: -x[1]
        ):
            doc = self._documents[idx].copy()
            doc["rrf_score"] = rrf_score
            doc["index"] = idx
            results.append(doc)

        return results

    # ── 持久化 ────────────────────────────────────────

    def save(self, path: str) -> None:
        """保存索引到磁盘。"""
        with open(path, "wb") as f:
            pickle.dump({
                "documents": self._documents,
                "doc_texts": self._doc_texts,
            }, f)
        print(f"  [Retriever] 已保存索引到 {path}")

    def load(self, path: str) -> None:
        """从磁盘加载索引。"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._documents = data["documents"]
        self._doc_texts = data["doc_texts"]
        self._build_bm25(self._doc_texts)
        print(f"  [Retriever] 已加载 {len(self._documents)} 个文档的索引")

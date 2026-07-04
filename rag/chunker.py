"""
RAG Chunker — 多种分块策略

安装依赖：
    uv pip install "sentence-transformers>=3.0.0" "rank-bm25>=0.2.2"
"""

import re
from typing import Optional


class FixedSizeChunker:
    """
    固定大小分块。

    最简单的分块策略：按字符数切，带 overlap。
    参数：
        chunk_size: 每块字符数（默认 512）
        overlap: 相邻块的重叠字符数（默认 50）
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[dict]:
        """将文本切成块，返回 [{text, index, start_char, end_char}]。"""
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # 尽量在句号/换行处断开
            if end < len(text):
                # 从 end 往前找最近的句号或换行
                cutoff = max(
                    text.rfind("。", start, end),
                    text.rfind("\n", start, end),
                    text.rfind(". ", start, end),
                )
                if cutoff > start + self.chunk_size // 2:
                    end = cutoff + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "index": idx,
                    "start_char": start,
                    "end_char": end,
                })
                idx += 1
            start = end - self.overlap if end < len(text) else len(text)

        return chunks


class SemanticChunker:
    """
    语义分块：按自然段落/标题分割，保留语义完整性。

    适用场景：Markdown 文档、有自然段落的文章。
    """

    def __init__(self, min_chunk_len: int = 100, max_chunk_len: int = 1500):
        self.min_chunk_len = min_chunk_len
        self.max_chunk_len = max_chunk_len

    def chunk(self, text: str) -> list[dict]:
        """按自然段落分块。"""
        # 按 Markdown 标题或双换行分割
        paragraphs = re.split(r"\n\s*\n|(?=^#+\s)", text.strip(), flags=re.MULTILINE)

        chunks = []
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para or len(para) < self.min_chunk_len:
                continue
            if len(para) > self.max_chunk_len:
                # 过长的段落递归做固定大小分块
                sub_chunker = FixedSizeChunker(
                    chunk_size=self.max_chunk_len, overlap=50
                )
                sub_chunks = sub_chunker.chunk(para)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    "text": para,
                    "index": i,
                    "start_char": -1,
                    "end_char": -1,
                })

        return chunks


class ParentChildChunker:
    """
    父子分块策略。

    Parent（大块）：用于检索时粗召回
    Child（小块）：精确匹配后返回给 LLM

    参数：
        parent_size: 父块字符数（默认 1024）
        child_size: 子块字符数（默认 256）
        overlap: 子块重叠（默认 30）
    """

    def __init__(
        self,
        parent_size: int = 1024,
        child_size: int = 256,
        child_overlap: int = 30,
    ):
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk(self, text: str) -> list[dict]:
        """
        返回父子结构列表。
        每个元素：{
            "parent": {"text": ..., "index": ...},
            "children": [{"text": ..., "index": ...}, ...]
        }
        """
        parent_chunker = FixedSizeChunker(
            chunk_size=self.parent_size, overlap=0
        )
        child_chunker = FixedSizeChunker(
            chunk_size=self.child_size, overlap=self.child_overlap
        )

        parents = parent_chunker.chunk(text)
        result = []
        for p in parents:
            children = child_chunker.chunk(p["text"])
            result.append({
                "parent": p,
                "children": children,
            })

        return result

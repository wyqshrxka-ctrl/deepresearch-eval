"""
Agent Memory 系统 — 短期 + 长期记忆

三种记忆类型：

┌──────────────────────────────────────────────────┐
│ 短期记忆 (Short-term)                             │
│ - 当前对话的历史消息                               │
│ - LangGraph 的 MessagesState 已提供               │
│ - 通过 MemorySaver checkpointer 持久化             │
│                                                   │
│ 长期记忆 (Long-term)                               │
│ - 跨会话的知识沉淀                                 │
│ - 用向量库存储实体信息                              │
│ - Agent 查询时自主检索                              │
│                                                   │
│ 工作记忆 (Working)                                 │
│ - 当前任务的中间状态                                │
│ - AgentState 的 plan / past_steps 字段已实现        │
└──────────────────────────────────────────────────┘

面试常见追问：
- Q: Short-term vs Working Memory 的区别？
  A: Short-term 是对话历史（用户说了什么），
     Working 是任务状态（当前计划、中间结果）。
     一个是"历史"，一个是"进度"。

- Q: 长期记忆怎么避免过时信息？
  A: 加时间戳降权 + TTL 过期 + Agent 判断可靠性。
"""

from typing import Optional
from langchain_core.messages import BaseMessage


class ShortTermMemory:
    """
    短期记忆：窗口缓冲，只保留最近 N 条消息。

    为什么用窗口而不是无限长？
    1. LLM context 有 token 上限（DeepSeek 64k，实际有效约 32k）
    2. 消息越多，LLM 越容易 lost in the middle——忘记中间说了什么
    3. 窗口内存占用可控，API 成本可控

    面试金句：
    "短期记忆用滑动窗口，通常保留最近 10-20 轮。
     超过窗口的消息会丢失，所以需要长期记忆来弥补。"
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns  # 最多保留最近 N 轮（1 轮 = user + ai）

    def trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """截断消息到最近 max_turns 轮，但总是保留第一条（system prompt）。"""
        if len(messages) <= self.max_turns * 2 + 1:
            return messages
        system = messages[:1]  # system prompt 不丢
        recent = messages[-(self.max_turns * 2):]
        return system + recent

    @property
    def stats(self) -> dict:
        return {"type": "short_term", "max_turns": self.max_turns}


class LongTermMemory:
    """
    长期记忆：用向量库存储实体和知识点。

    设计考虑：
    - 用 Chroma 做存储（轻量，不需要额外服务）
    - 每个记忆条目有：文本内容、类型（fact/user_preference）、时间戳
    - 检索时用余弦相似度召回 top-k

    面试金句：
    "长期记忆我用向量库实现，实体和偏好分开存。
     每次 Agent 执行任务时，先检索历史相关知识，
     和当前上下文一起注入 prompt。"
    """

    def __init__(self, collection_name: str = "agent_memory"):
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    @property
    def client(self):
        """延迟初始化 Chroma 客户端。"""
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(
                path="./chroma_data"
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name
            )
        return self._client

    @property
    def collection(self):
        self.client  # 触发初始化
        return self._collection

    def add(self, content: str, metadata: Optional[dict] = None) -> str:
        """
        写入一条长期记忆。

        返回记忆 ID，可以用于删除或更新。
        """
        import uuid
        import time

        mem_id = str(uuid.uuid4())[:8]
        meta = {
            "timestamp": time.time(),
            "type": "general",
        }
        meta.update(metadata or {})

        self.collection.add(
            ids=[mem_id],
            documents=[content],
            metadatas=[meta],
        )
        return mem_id

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """
        检索与 query 相关的长期记忆。

        返回 top_k 条记忆的文本内容。
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        if results and results["documents"] and results["documents"][0]:
            return results["documents"][0]
        return []

    def forget(self, mem_id: str) -> None:
        """删除一条记忆。"""
        self.collection.delete(ids=[mem_id])

    def list_all(self) -> list[dict]:
        """列出所有记忆。"""
        results = self.collection.get()
        items = []
        if results and results["ids"]:
            for mem_id, doc, meta in zip(
                results["ids"], results["documents"] or [], results["metadatas"] or []
            ):
                items.append({"id": mem_id, "content": doc, "metadata": meta})
        return items

    @property
    def stats(self) -> dict:
        return {
            "type": "long_term",
            "count": self.collection.count(),
            "backend": "Chroma (persistent)",
        }

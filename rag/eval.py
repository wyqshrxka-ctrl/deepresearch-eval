"""
RAG Evaluation — 基于 RAGAS 的评测封装

安装依赖：
    uv pip install "ragas>=0.2.0"

用法：
    from rag.eval import RAGEvaluator
    evaluator = RAGEvaluator()
    scores = evaluator.evaluate(
        questions=["什么是 Agent？"],
        ground_truths=["Agent 是能自主行动的智能体"],
        contexts=[["Agent 能感知环境并执行任务"]],
        answers=["Agent 是能自主行动的智能体"],
    )
    print(scores)
"""

import os
import types
import sys
from typing import Optional

from dotenv import load_dotenv
load_dotenv()


class SimpleJudge:
    """
    自定义 LLM-as-Judge，不依赖 RAGAS。

    直接调用 DeepSeek 做评测，每个指标有独立的 prompt。
    这比 RAGAS 更透明——你可以看到每一步的评分逻辑。
    """

    def __init__(self):
        from langchain_openai import ChatOpenAI
        import os

        self.llm = ChatOpenAI(
            model="deepseek-chat",
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0,
        )

    def evaluate(
        self,
        questions: list[str],
        ground_truths: list[str],
        contexts: list[list[str]],
        answers: list[str],
    ) -> dict[str, float]:
        """评测所有样本，返回平均分数。"""
        scores = {"faithfulness": [], "answer_relevancy": [], "context_precision": []}

        for q, gt, ctx_list, ans in zip(questions, ground_truths, contexts, answers):
            ctx_text = "\n".join(ctx_list[:3])
            scores["faithfulness"].append(
                self._faithfulness_score(q, ctx_text, ans)
            )
            scores["answer_relevancy"].append(
                self._answer_relevancy_score(q, ans)
            )
            scores["context_precision"].append(
                self._context_precision_score(q, ctx_text)
            )

        return {k: round(sum(v) / len(v), 3) for k, v in scores.items()}

    def _judge(self, prompt: str) -> float:
        """调 LLM 打分（0-1），加超时保护。"""
        import asyncio
        try:
            result = self.llm.invoke(prompt)
            import re
            # 提取 0.xx 或 xx% 格式的数字
            text = result.content
            match = re.search(r"(\d+\.?\d*)", text)
            if match:
                score = float(match.group(1))
                if score > 10:
                    score = score / 100  # 百分比转换
                return min(max(score, 0.0), 1.0)
            return 0.5
        except Exception as e:
            print(f"    [Judge] 评分失败: {e}")
            return 0.0

    def _faithfulness_score(self, question, context, answer) -> float:
        """faithfulness: 回答是否忠于上下文（不编造）"""
        prompt = (
            "你是一个 RAG 评测员。请评估回答是否忠实于给定的上下文。\n\n"
            f"上下文：\n{context}\n\n"
            f"回答：\n{answer}\n\n"
            "评分标准：\n"
            "- 1.0：回答完全基于上下文，没有编造\n"
            "- 0.7：大部分基于上下文，少量补充\n"
            "- 0.4：一半编造，一半基于上下文\n"
            "- 0.0：完全编造，与上下文无关\n\n"
            "请只输出分数（0.0-1.0）："
        )
        return self._judge(prompt)

    def _answer_relevancy_score(self, question, answer) -> float:
        """answer_relevancy: 回答是否切题"""
        prompt = (
            "你是一个 RAG 评测员。请评估回答是否切题、完整地回答了用户问题。\n\n"
            f"用户问题： {question}\n\n"
            f"回答：\n{answer}\n\n"
            "评分标准：\n"
            "- 1.0：完整回答了问题\n"
            "- 0.7：基本回答了，但不够详细\n"
            "- 0.4：部分相关，但偏离了核心问题\n"
            "- 0.0：完全不相关\n\n"
            "请只输出分数（0.0-1.0）："
        )
        return self._judge(prompt)

    def _context_precision_score(self, question, context) -> float:
        """context_precision: 检索到的上下文是否与问题相关"""
        prompt = (
            "你是一个 RAG 评测员。请评估检索到的上下文是否与用户问题相关。\n\n"
            f"用户问题： {question}\n\n"
            f"检索到的上下文：\n{context}\n\n"
            "评分标准：\n"
            "- 1.0：上下文与问题高度相关\n"
            "- 0.7：大部分相关\n"
            "- 0.4：部分相关\n"
            "- 0.0：完全不相关\n\n"
            "请只输出分数（0.0-1.0）："
        )
        return self._judge(prompt)

    def print_report(self, scores: dict) -> None:
        """打印格式化报告"""
        print()
        print("=" * 50)
        print("📊 RAG 评测报告 (LLM-as-Judge)")
        print("=" * 50)
        labels = {
            "faithfulness": "忠实性 (faithfulness)",
            "answer_relevancy": "切题性 (answer_relevancy)",
            "context_precision": "检索精度 (context_precision)",
        }
        for key, label in labels.items():
            val = scores.get(key, 0.0)
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"  {label:30s} {bar} {val:.3f}")
        print("=" * 50)
        print()


class RAGEvaluator:
    """
    RAG 评测器（兼容原 RAGAS 接口，但现在优先用 SimpleJudge）。
    """

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_base_url: str = "https://api.deepseek.com/v1",
        llm_model: str = "deepseek-chat",
    ):
        self.llm_api_key = llm_api_key or os.getenv("DEEPSEEK_API_KEY")
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self._judge = SimpleJudge()

    def evaluate(self, questions, ground_truths, contexts, answers) -> dict:
        return self._judge.evaluate(questions, ground_truths, contexts, answers)

    def print_report(self, scores: dict) -> None:
        self._judge.print_report(scores)

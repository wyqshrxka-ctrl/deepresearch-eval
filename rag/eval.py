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
from typing import Optional


class RAGEvaluator:
    """
    RAG 评测器，基于 RAGAS 框架。

    支持三个核心指标：
    - faithfulness: 回答是否忠于检索到的上下文（不编造）
    - answer_relevancy: 回答是否回答了问题
    - context_precision: 检索到的上下文是否相关

    RAGAS 默认使用 OpenAI，这里通过环境变量配成 DeepSeek。
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

        # 设置环境变量供 RAGAS 使用
        if self.llm_api_key:
            # RAGAS 兼容 OpenAI 接口，设置成 DeepSeek
            os.environ["OPENAI_API_KEY"] = self.llm_api_key
            # RAGAS 默认用 OpenAI 的 chat 端点，这里改 base_url
            from ragas.llms import llm_factory
            from langchain_openai import ChatOpenAI

            self._judge_llm = ChatOpenAI(
                model=self.llm_model,
                base_url=self.llm_base_url,
                api_key=self.llm_api_key,
                temperature=0,
            )
            # 注入到 ragas 的全局 LLM
            # NOTE: ragas 版本更新后 API 可能变化，这里做兼容
            try:
                from ragas import evaluate as ragas_evaluate
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                )
                self.ragas_evaluate = ragas_evaluate
                self.metrics = [faithfulness, answer_relevancy, context_precision]
            except ImportError:
                print("  [Eval] ragas 未安装，请运行:")
                print("    uv pip install 'ragas>=0.2.0'")
                self.ragas_evaluate = None

    def evaluate(
        self,
        questions: list[str],
        ground_truths: list[str],
        contexts: list[list[str]],
        answers: list[str],
    ) -> dict:
        """
        运行 RAGAS 评测。

        参数：
            questions: 问题列表
            ground_truths: 标准答案列表
            contexts: 检索到的上下文（每条问题对应一个列表）
            answers: 系统生成的回答列表

        返回：
            {"faithfulness": 0.85, "answer_relevancy": 0.72, ...}
        """
        if self.ragas_evaluate is None:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "error": "ragas 未安装",
            }

        from datasets import Dataset

        data = {
            "question": questions,
            "ground_truth": ground_truths,
            "contexts": contexts,
            "answer": answers,
        }
        dataset = Dataset.from_dict(data)

        try:
            scores = self.ragas_evaluate(
                dataset,
                metrics=self.metrics,
                llm=self._judge_llm,
            )
            return scores.to_pandas().to_dict("records")[0]
        except Exception as e:
            print(f"  [Eval] 评测失败: {e}")
            return {"error": str(e)}

    def print_report(self, scores: dict) -> None:
        """打印格式化的评测报告。"""
        print()
        print("=" * 50)
        print("📊 RAG 评测报告")
        print("=" * 50)
        for metric, value in scores.items():
            if metric == "error":
                continue
            if isinstance(value, float):
                bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
                print(f"  {metric:25s} {bar} {value:.3f}")
            else:
                print(f"  {metric:25s} {value}")
        print("=" * 50)
        print()

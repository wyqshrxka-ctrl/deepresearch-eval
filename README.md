# DeepResearch Eval

> A multi-agent deep research system with full evaluation pipeline.
> Built as a portfolio for 2026 校招 Agent 工程师 positions.

## Overview

This project demonstrates a production-grade Agent system covering:

- 🤖 **Multi-Agent Orchestration** — Planner / Searcher / Reader / Writer / Critic
- 🛠️ **MCP-based Tooling** — Standardized tool interface via Model Context Protocol
- 📚 **Hybrid RAG** — BM25 + dense + reranker, configurable per query
- 📊 **Evaluation Layer** — GAIA / FRAMES / custom benchmark with LLM-as-Judge
- 🔍 **Observability** — Langfuse tracing, token cost, error attribution
- 🧪 **Reproducibility** — Ablation studies, multi-model comparison

## Quick Start

```bash
# 1. 安装依赖
make install

# 2. 配置 API key
cp .env.example .env
# 编辑 .env 填入你的 key

# 3. 启动 MCP server（终端 1）
make run-server

# 4. 测试 MCP client（终端 2）
make run-client

# 5. 跑 Plan-and-Execute Agent（终端 3）
make run-agent
```

## Architecture

```
┌─────────────────────────────────────────┐
│        Plan-and-Execute Agent           │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │ Planner  │→ │ Executor │→ │Replan- │││
│  │   LLM    │  │   Loop   │  │  ner   ││
│  └──────────┘  └────┬─────┘  └────────┘│
└────────────────────┼────────────────────┘
                     │ MCP Protocol
        ┌────────────┴────────────┐
        │      MCP Server         │
        │  - search_web           │
        │  - fetch_url            │
        │  - calculator           │
        └─────────────────────────┘
```

## Project Structure

```
deepresearch-eval/
├── agent/              # Agent 核心
│   └── plan_execute.py # Plan-and-Execute 实现
├── mcp_server/         # MCP 工具服务端
│   └── main.py         # 暴露 search/fetch/calc 工具
├── mcp_client/         # MCP 客户端 demo
│   └── demo.py
├── tests/              # 单元测试
└── docs/               # 设计文档
```

## Roadmap

- [x] **W1**: LangGraph + MCP basics, Plan-and-Execute v0
- [ ] **W2**: Hybrid RAG with BM25 + Reranker, RAGAS eval
- [ ] **W3**: LLM 原理沉淀 + 模拟面试
- [ ] **W4-7**: GAIA Lv1 baseline + ablation
- [ ] **W8**: 50 题中文评测集 + LLM-as-Judge + Cohen's κ
- [ ] **W9**: Langfuse 可观测
- [ ] **W10**: LoRA SFT 浅尝
- [ ] **W11-12**: 博客 + 简历 + mock interview

## License

MIT

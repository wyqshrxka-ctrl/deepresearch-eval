"""
用于测试 RAG 的小型文档集（10 篇，主题：Agent / 评测 / LLM）。

每篇文档格式：{"title": "...", "content": "..."}
"""

TEST_DOCUMENTS = [
    {
        "title": "AI Agent 的定义与核心特征",
        "content": (
            "AI Agent（智能体）是指能够自主感知环境、做出决策并执行任务的智能系统。"
            "与传统程序不同，Agent 具备目标导向性、自主性和交互能力。"
            "一个典型的 Agent 包含感知模块（接收输入）、推理模块（规划决策）和执行模块（调用工具）。"
            "Agent 可以是单轮问答的简单形式，也可以是多轮交互、多工具调用的复杂系统。"
            "近年来，基于大语言模型的 Agent 成为研究热点，LLM 作为 Agent 的「大脑」理解用户意图、规划任务步骤。"
        ),
    },
    {
        "title": "Agent 评测方法论",
        "content": (
            "Agent 评测是衡量智能体系统效果的关键环节。评测方法分为离线评测和在线评测两大类。"
            "离线评测使用预定义测试集，评估 Agent 在不同任务上的准确率、完成率和效率。"
            "在线评测则在真实用户场景中收集反馈数据，关注用户满意度、任务完成时长等指标。"
            "常见的 Agent 评测框架包括 GAIA（通用 Agent 推理）、FRAMES（检索增强）、SWE-bench（代码 Agent）。"
            "LLM-as-Judge 是一种新兴的评测方法：让大模型作为裁判，评估 Agent 输出的质量。"
            "但 LLM-as-Judge 存在 position bias、length bias、self-preference 等偏置，需要校准。"
        ),
    },
    {
        "title": "LLM-as-Judge 偏置校准方法",
        "content": (
            "LLM-as-Judge 是一种用大模型自动评测输出的方法。它高效但存在多种偏置。"
            "第一种是 position bias：Judge 倾向于选择第一个或最后一个答案。"
            "第二种是 length bias：Judge 倾向于给更长的回答更高分。"
            "第三种是 self-preference：Judge 更偏好和自己风格相似的答案。"
            "第四种是 verbosity bias：回答越啰嗦，Judge 越倾向给高分。"
            "校准方法包括：多次交换选项顺序取平均值、加入参考答案、和人工标注计算 Cohen's κ。"
            "Cohen's κ 是衡量两个评估者一致性的指标，κ > 0.7 表示一致性良好。"
        ),
    },
    {
        "title": "RAG 检索增强生成技术",
        "content": (
            "RAG（Retrieval-Augmented Generation）是解决大模型知识更新和幻觉问题的核心技术。"
            "RAG 的基本流程：用户查询 → 检索相关内容 → 将检索结果注入 Prompt → LLM 生成回答。"
            "检索阶段通常使用向量相似度搜索：将文档和查询编码为 Embedding 向量，计算余弦相似度。"
            "Hybrid Search 结合了 BM25（关键词精确匹配）和向量检索（语义匹配），效果更好。"
            "Reranker 是 RAG 中的精排环节：先用粗检召回 top 50，再用 cross-encoder 精排到 top 5。"
            "Reranker 虽然慢但精度高，适合放在检索链的最后一步。"
        ),
    },
    {
        "title": "Agent Memory 系统设计",
        "content": (
            "Memory 是 Agent 持续学习和跨轮交互的核心能力。Memory 系统分为三个层次。"
            "短期记忆（Short-term Memory）存储当前会话的对话历史，通常用 buffer 或窗口实现。"
            "长期记忆（Long-term Memory）存储跨会话的知识和经验，通常用向量数据库实现。"
            "工作记忆（Working Memory）存储当前任务中的中间状态和工具调用结果。"
            "Memory 的设计需要考虑：存储容量、检索速度、相关性排序和遗忘机制。"
            "LangGraph 提供了 MemorySaver 用于短期记忆持久化，Checkpointer 用于状态快照。"
        ),
    },
    {
        "title": "多 Agent 编排模式",
        "content": (
            "多 Agent 编排是解决复杂任务的关键模式。常见的编排模式有以下几种。"
            "Supervisor 模式：一个中央 Supervisor Agent 路由任务给多个 Worker Agent，Worker 不直接通信。"
            "Plan-and-Execute 模式：先由 Planner 生成完整计划，再逐步执行，中间由 Replanner 调整计划。"
            "Hierarchical 模式：多层级的 Agent 组织，上层 Agent 分解任务给下层 Agent。"
            "Swarm 模式：多个 Agent 对等协作，通过共享消息总线通信。"
            "选择哪种模式取决于任务的复杂度、子任务间的依赖关系和错误容忍度。"
        ),
    },
    {
        "title": "MCP 协议与工具标准化",
        "content": (
            "MCP（Model Context Protocol）是 Anthropic 提出的 LLM 工具调用标准化协议。"
            "MCP 包含三个核心抽象：Tool（LLM 主动调用的工具）、Resource（Client 提供给 LLM 的上下文）、Prompt（可复用模板）。"
            "MCP Server 通过 JSON-RPC 2.0 协议暴露能力，支持 stdio 和 SSE 两种传输方式。"
            "MCP 解决了 N×M 的适配问题：以前每个 LLM 应用要单独对接每个工具，现在工具只要实现 MCP Server 即可。"
            "2025 年 MCP 已成为事实标准，OpenAI、DeepSeek、豆包等主流平台都已支持。"
        ),
    },
    {
        "title": "Prompt Engineering 最佳实践",
        "content": (
            "Prompt Engineering 是 LLM 应用开发的基础技能。好的 Prompt 能显著提升 Agent 的表现。"
            "Few-shot 示例是最高效的 Prompt 技术之一：在 Prompt 中加入 2-3 个输入输出示例。"
            "Chain-of-Thought（思维链）引导 LLM 逐步推理，在复杂推理任务上效果显著。"
            "System Prompt 应该包含角色定义、行为约束、输出格式和边界条件。"
            "工具描述要写得像 Prompt：指明什么时候用、参数含义、返回格式和典型示例。"
            "工具数量建议控制在 10 个以内，否则 LLM 选错工具的几率会显著上升。"
        ),
    },
    {
        "title": "自省型 RAG：Self-RAG 与 Corrective RAG",
        "content": (
            "Self-RAG 和 Corrective RAG 是两种让 RAG 系统具备自省能力的改进方案。"
            "Self-RAG 的核心：LLM 在生成过程中自行判断'是否需要检索'以及'检索结果是否可信'。"
            "它通过特殊的 token 控制检索时机，避免不必要的 API 调用。"
            "Corrective RAG（CRAG）在检索后增加一个评估步骤：如果检索结果质量差，则重写查询重新检索。"
            "CRAG 还会对检索结果进行去噪和重排，只保留最相关的内容给 LLM。"
            "两者的本质区别：Self-RAG 判断'要不要检'，CRAG 判断'检得好不好'。"
        ),
    },
    {
        "title": "GraphRAG 与知识图谱增强检索",
        "content": (
            "GraphRAG 是微软提出的一种将知识图谱引入 RAG 的方法。"
            "传统 RAG 将文档切分为独立片段，丢失了片段之间的实体关系和层级结构。"
            "GraphRAG 先对文档进行实体抽取和关系建模，构建知识图谱。"
            "检索时，GraphRAG 不仅检索相关文本片段，还检索图谱中的实体和关系路径。"
            "这使得 GraphRAG 在处理多跳推理问题时表现更好。"
            "例如'某公司的创始人的出生地'——需要关联两跳信息，GraphRAG 能直接沿图谱找到答案。"
            "GraphRAG 的缺点是建图成本高、更新复杂，适合静态知识库。"
        ),
    },
]

"""
W3 — 15 个原理点面试精简回答

这份文件是 W3 的核心产出。每个原理点控制在 50 字以内，面试官追问时不卡壳。

运行：
    .venv/bin/python notebooks/08_llm_principles.py

产出：
    1. 15 个原理点精简回答（打印到终端，可直接截图复习）
    2. 7 个面试核心追问 + 标准答案
    3. W3 自测打分
"""

print("=" * 60)
print("🧠 W3 — LLM 原理 15 点精简回答")
print("=" * 60)

principles = {
    # ── Agent 类（4 个）──
    "1. ReAct vs Plan-and-Execute vs ReWOO": (
        "ReAct 一次只想一步（观察→思考→行动→观察），灵活但短视。"
        "Plan-and-Execute 先全局规划再分步执行，全局视野但容错差。"
        "ReWOO 不记录观察，通过推理链高效调用——省钱但可靠性低。"
    ),
    "2. Function Calling 底层怎么工作": (
        "本质是两件事：1) 在 System Prompt 里注入工具 schema（JSON 格式），"
        "2) LLM 输出特殊 token 触发工具调用。不是 LLM 真的'执行'了代码，"
        "是框架拦截 JSON、调函数、把结果塞回 prompt。"
    ),
    "3. 什么时候需要多 Agent": (
        "单 Agent 擅长独立任务，多 Agent 擅长需要角色分工的场景（研究+写作+审查）。"
        "关键判断：一个 Agent 做全部能不能保持 quality？能就单，不能就多。"
        "多 Agent 的代价是通信开销和编排复杂度。"
    ),
    "4. Memory 的三种类型": (
        "Short-term = 对话历史（最近说了什么）；Working = 任务状态（正在做什么）；"
        "Long-term = 跨会话知识（以前讨论过什么）。Short-term 窗口截断，"
        "Long-term 向量检索，Working 与 State 耦合。"
    ),

    # ── RAG 类（4 个）──
    "5. Embedding 为什么能表示语义": (
        "Embedding 把词映射到高维空间的向量。语义相近的词（如 '猫' 和 '狗'）"
        "在向量空间中距离近（余弦相似度高）。不是魔法，是大规模语料训练的结果。"
    ),
    "6. 为什么需要 Reranker": (
        "Bi-encoder（Embedding 模型）检索快但精度低（query 和 doc 独立编码），"
        "Cross-encoder（如 bge-reranker）把 query+doc 一起编码，精度高但慢。"
        "所以：先粗检用 bi-encoder，再精排用 cross-encoder。"
    ),
    "7. Hybrid Search 的本质": (
        "BM25 抓关键词（召回'精确提到这个词'的文档），"
        "Dense 抓语义（召回'说的是一回事'的文档）。两者互补。"
        "RRF 按排名融合：不管分值差异多大，只看排名。"
    ),
    "8. Chunking 策略对效果的影响": (
        "太大：塞入无关信息，LLM 分心；太小：切碎语义，丢失上下文。"
        "固定大小适合 FAQ，语义切分适合长文，父子分块适合需要引用的场景。"
        "面试常问：'你 chunk size 怎么定的？'答'256 token 是经验值，视文档类型调'。"
    ),

    # ── 评测类（4 个）──
    "9. LLM-as-Judge 的 4 种 bias": (
        "1) Position bias：倾向于选第一个/最后一个选项。校准：交换选项顺序取平均。"
        "2) Length bias：倾向于长的回答。校准：加长度惩罚或截断。"
        "3) Self-preference：偏好自己风格。校准：用不同的 LLM 做 Judge。"
        "4) Verbosity bias：啰嗦的得分高。校准：要求只输出分数。"
    ),
    "10. Cohen's κ 怎么验证 Judge": (
        "Cohen's κ 衡量两个评估者（Judge vs 人工）在分类或评分上的一致性。"
        "κ > 0.6 表示中等一致，κ > 0.8 表示强一致。"
        "高于 0.7 通常被认为可靠——这是面试时你要记住的数字。"
    ),
    "11. RAG 的三个核心指标": (
        "Faithfulness：回答是否忠于上下文（不编造）。"
        "Answer Relevancy：回答是否切题（回答了用户问的问题）。"
        "Context Precision：检索到的上下文是否与问题相关（没搜偏）。"
        "三个指标对应 RAG 链路的三个断点：检索、理解、生成。"
    ),
    "12. 消融实验的设计原则": (
        "控制变量法：每次只改变一个组件（去掉 Reranker / 换 Embedding 模型 / 改 chunk size），"
        "其他条件不变。多次采样取平均（至少 3 次），消随机性。"
        "文档化每一次实验的参数和结论——面试官看这个判断你的实验思维。"
    ),

    # ── Transformer 原理类（3 个）──
    "13. Transformer 的核心组件": (
        "四个关键组件：1) Attention（'哪些词和我相关'），"
        "2) FFN（把 Attention 输出加工成更丰富的表示），"
        "3) Residual Connection（防止梯度消失，原输入 + 新输出），"
        "4) LayerNorm（标准化每一层的输出，让训练稳定）。"
        "记住——Attention 是'看'，FFN 是'想'，Residual 是'不忘'。"
    ),
    "14. Decoding 策略对比": (
        "Greedy = 每次选最高概率 token，快但容易重复。"
        "Top-k = 只从概率最高的 k 个里选，增加多样性。"
        "Top-p = 从累积概率 > p 的集合里选，自适应。"
        "Temperature = 调概率分布：T<1 保守，T>1 放飞。T=0 时结果确定。"
    ),
    "15. Context Window 的 lost in the middle 现象": (
        "LLM 对文档开头和结尾的信息记得最好，中间部分容易丢。"
        "所以长文档优先放在开头或结尾。128k context 不等于真的能用 128k。"
        "实际有效 context 往往是标示值的一半——面试官如果自己没做过长上下文应用，这句话会镇住他。"
    ),
}

for key, value in principles.items():
    print(f"\n{'─' * 60}")
    print(f"  {key}")
    print(f"{'─' * 60}")
    print(f"  {value}")


# ── 7 个核心追问 ──────────────────────────────────
print("\n\n" + "=" * 60)
print("🎯 面试 7 个核心追问（你答一遍）")
print("=" * 60)

interview_qa = [
    ("Transformer 的核心组件是哪几个？", (
        "Attention（看哪些信息相关）+ FFN（加工信息）+ "
        "Residual（保留原始信息）+ LayerNorm（稳定训练）。"
        "可以用人话讲：Attention 是'看'，FFN 是'想'，Residual 是'不忘'。"
    )),
    ("Self-Attention 为什么叫 self？", (
        "因为 query、key、value 都来自同一个序列——"
        "句子里的每个词看同一个句子里的其他词，自己看自己。"
        "对比 Cross-Attention：encoder 看 decoder，跨不同序列。"
    )),
    ("RAG 为什么能减少幻觉？", (
        "幻觉的根因是 LLM 靠概率生成文本，没有事实锚点。"
        "RAG 在生成前先检索外部知识，把事实注入 prompt，"
        "相当于给 LLM '开卷考试'。"
        "但 RAG 不能 100% 消除幻觉——如果检索结果本身就是错的，LLM 照样信。"
    )),
    ("Context Window 越大越好吗？", (
        "不是。越大越贵（API 按 token 计费），而且有 lost in the middle 现象。"
        "实际有效 context 大约是指标值的一半。"
        "选择 window 的策略：按最长的真实输入定义，不要为了炫耀选大的。"
    )),
    ("temperature=0 和 temperature=1 的区别？", (
        "T=0：每次都是最高概率 token，结果确定。适合正式任务（代码生成、翻译）。"
        "T=1：按原始概率分布采样，有随机性。适合创意任务（故事生成）。"
        "T>1：概率更平均，结果放飞。一般不推荐。"
    )),
    ("LLM 内部是怎么处理输入文本的？", (
        "Tokenize → 查表转成 Embedding 向量 → 送进 Transformer 层 → "
        "每层做 Attention + FFN → 最后一层输出 logits → "
        "Softmax 转成概率分布 → 采样生成下一个 token。"
        "面试时画个简单的流程图就够了。"
    )),
    ("什么样的问题不适合用 LLM 解决？", (
        "1) 需要精确计算的（LLM 不擅长数学，请用 calculator tool）。"
        "2) 需要实时数据的（LLM 有知识截止日）。"
        "3) 需要确定性输出的（LLM 本质上是概率生成，有随机性）。"
        "解决办法是 tool use——把计算 / 搜索 / 数据库查询外挂成 function calling。"
    )),
]

for q, a in interview_qa:
    print(f"\n{'─' * 60}")
    print(f"  Q: {q}")
    print(f"{'─' * 60}")
    print(f"  A: {a}")


# ── W3 自测 ───────────────────────────────────────
print("\n\n" + "=" * 60)
print("📝 W3 自测打分（满分 22）")
print("=" * 60)
print("""
  原理 15 题:  __/15（每题 1 分，能在 30 秒内用人话讲出来就得分）
  面试 7 题:   __/7（每题 1 分，能讲清楚 + 举例就得分）

  总分: __/22
""")

print("✅ W3 原理点速通完成")
print("   保存这份打印输出到笔记 / 截图 / 纸笔抄一遍，面试前过一遍。")

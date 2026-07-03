# Day 2 上手指南

> 你周二打开就能跟着做。整套预计 3-4 小时。

---

## Step 0：环境准备（10 分钟）

### 0.1 把脚手架变成你的项目

```bash
# 1. 把整个 scaffold 复制到一个新位置（这才是你的真实仓库）
cp -r /Users/wwwyyyqqq/Documents/test/deepresearch-eval-scaffold ~/code/deepresearch-eval
cd ~/code/deepresearch-eval

# 2. 初始化 git
git init
git add .
git commit -m "chore: project scaffold"

# 3. 在 GitHub 建空仓 deepresearch-eval（别勾选 README，否则要 merge）
# 然后：
git remote add origin git@github.com:你的用户名/deepresearch-eval.git
git branch -M main
git push -u origin main
```

### 0.2 装 uv（如果还没装）

```bash
brew install uv
# 或
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 0.3 装依赖

```bash
make install
```

### 0.4 配置 API key

```bash
cp .env.example .env
# 编辑 .env：
# - 至少配 DEEPSEEK_API_KEY（去 https://platform.deepseek.com 注册，10 元够用一周）
# - 配 TAVILY_API_KEY（去 https://tavily.com 注册免费 1000 次/月）
```

---

## Step 1：跑 LangGraph Tutorial 1 - Quick Start（1 小时）

**目标**：理解 State + Node + Edge

### 1.1 跟着官方文档跑
打开 https://langchain-ai.github.io/langgraph/tutorials/get-started/

**别复制粘贴**，自己一行一行敲（重点是跑通 + 理解，不是速度）。

### 1.2 跑通后做这件事

把 tutorial 的代码改造一下：
- 加一个新工具（比如查日期 `get_today()`）
- 把图改成：先调日期工具 → 再调搜索 → 输出

### 1.3 提交

```bash
mkdir -p notebooks
# 把 tutorial 的代码保存到 notebooks/01_langgraph_quickstart.py
git add notebooks/01_langgraph_quickstart.py
git commit -m "feat: langgraph quickstart with custom tool"
git push
```

### 1.4 自检 3 个问题（写到 notebooks/01_*.py 顶部注释里）
- State 用 TypedDict 而不是 Pydantic 是为什么？
- `Annotated[list, add_messages]` 不写 `add_messages` 会怎样？
- `graph.compile()` 做了什么？

---

## Step 2：跑 LangGraph Tutorial 2 - Multi-Agent Supervisor（1 小时）

**目标**：理解 Orchestrator-Workers 模式

### 2.1 跟着官方文档跑
https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/

### 2.2 跑通后做这件事

加一个 Critic Agent：在 Writer 写完后，让 Critic 检查回答质量，不满意就让 Writer 重写。

### 2.3 提交

```bash
git add notebooks/02_langgraph_supervisor.py
git commit -m "feat: multi-agent supervisor + critic"
git push
```

### 2.4 自检 3 个问题
- Supervisor 是怎么决定下一个 Agent 的？
- 子 Agent 之间能直接通信吗？
- 如果一个子 Agent 卡住怎么办？

---

## Step 3：跑 LangGraph Tutorial 3 - Plan-and-Execute（1 小时）

**目标**：理解 Plan-and-Execute 范式（W1 收官项目用得上）

### 3.1 跟着官方文档跑
https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/

### 3.2 跑通后做这件事

用一句话写下：Plan-and-Execute 比 ReAct 强在哪？什么场景反而更差？

### 3.3 提交

```bash
git add notebooks/03_langgraph_plan_execute.py
git commit -m "feat: plan-and-execute tutorial"
git push
```

---

## Step 4：写 W1 Day 2 周记（10 分钟）

新建 `notes/day2.md`：

```markdown
# Day 2 周二（2026-06-03）

## 跑通了什么
- [x] LangGraph Tutorial 1 - Quick Start
- [x] LangGraph Tutorial 2 - Multi-Agent Supervisor
- [x] LangGraph Tutorial 3 - Plan-and-Execute

## 最有感触的 1 个点
（写一句话）

## 踩了什么坑
- 坑：xxx → 解决：xxx

## 不懂的地方
- xxx
```

---

## 常见问题

### Q1：tutorial 报错 `ImportError: cannot import name 'xxx'`
A：版本问题。LangGraph API 变化频繁。建议固定版本：
```bash
pip install langgraph==0.2.50 langchain==0.3.10
```

### Q2：DeepSeek API 调不通
A：检查：
- `.env` 里 `DEEPSEEK_BASE_URL` 是否是 `https://api.deepseek.com/v1`
- key 是否有效（去后台看 quota）
- `model` 名是否是 `deepseek-chat`（v3）或 `deepseek-reasoner`（r1）

### Q3：tutorial 用 OpenAI 我没 key
A：把 `ChatOpenAI(model="gpt-4o-mini")` 全部替换成：
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key="你的key",
)
```

### Q4：可视化图 `draw_mermaid_png()` 报错
A：那个需要 Jupyter + 网络。跳过即可，或者用 `print(app.get_graph().draw_ascii())`。

---

## Day 3-6 的预告

- **Day 3**：把 tutorial 学到的东西用到自己的 `agent/plan_execute.py` 上，先跑通骨架
- **Day 4**：填 `mcp_server/main.py` 里的 `calculator` TODO
- **Day 5**：跑通 mcp_server + mcp_client 联调
- **Day 6**：填 `agent/plan_execute.py` 里的 `execute_with_tools` TODO，跑通完整链路
- **Day 7**：W1 周记 + 自测

每天 commit 至少 1 次。**节奏感比代码漂亮重要 10 倍**。

.PHONY: install dev fmt lint test clean run-server run-client run-agent

# === 环境 ===

install:
	uv venv
	uv pip install -e ".[dev]"

dev: install
	@echo "✅ 环境就绪，请复制 .env.example 到 .env 并填入 API keys"

# === 代码质量 ===

fmt:
	ruff format .
	ruff check --fix .

lint:
	ruff check .
	mypy agent mcp_server mcp_client

test:
	pytest tests/ -v

# === 运行 ===

# Day 5：跑 MCP server（在另一个终端跑 client）
run-server:
	python -m mcp_server.main

# Day 5：跑 MCP client，连接 server，列出工具并测试
run-client:
	python -m mcp_client.demo

# Day 6：跑 Plan-and-Execute Agent，调用 MCP server 完成任务
run-agent:
	python -m agent.plan_execute "今天北京的天气怎么样，帮我推荐穿搭"

# === 清理 ===

clean:
	rm -rf .venv __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ragpulse

轻量级 RAG + Agent 工程（参考 RAGFlow 思路裁剪）：**解析 → 向量化 → Chroma → Top‑K 检索 →（可选）LLM/Agent 生成**。  
目标：去掉 Infinity / ES / 完整 RAGFlow 后端后，仍能本地跑通并便于二次开发。

## 已实现能力

- **入库**：多格式解析 + 分块 + embedding + 写入 Chroma（同步/异步）
- **检索**：查询 embedding + Chroma Top‑K（命中支持导出 JSON）
- **RAG QA**：检索片段拼上下文 → LLM 回答（可选多模态图片 URL）
- **Agent**：`/agent/chat` 支持会话上下文、长期记忆、可选 RAG
- **Web UI**：浏览器上传入库、问答、命中列表（含命中图片缩略图）、Agent 会话

## 快速开始

### 1) 安装与配置

```bash
cd ragpulse
pip install -r requirements.txt
cp .env.example .env
```

至少配置（见 `.env.example`）：

- `EMBEDDING_API_KEY`（或复用 `LLM_API_KEY`）
- `LLM_API_KEY` / `LLM_API_URL` / `LLM_MODEL`（用于 `/rag/qa` 或 Agent 生成）

### 2) 启动服务（含 Web + Agent）

```bash
PYTHONPATH=. uvicorn main:app --host 0.0.0.0 --port 8000
```

- **Web**：`http://127.0.0.1:8000/`
- **健康检查**：`GET /health`

### 3) 常用 API

- **入库（异步，推荐）**：`POST /rag/ingest/async` → `GET /rag/ingest/tasks/{task_id}`
- **仅检索**：`POST /rag/retrieve`
- **RAG 问答**：`POST /rag/qa`
- **Agent 对话**：`POST /agent/chat`
- **会话列表**：`GET /agent/sessions/list?memory_id=...&user_id=...`
- **记忆概览**：`GET /agent/memory/list?memory_id=...&user_id=...`
- **清理记忆**：`POST /agent/memory/clear`

示例（Agent）：

```bash
curl -X POST "http://127.0.0.1:8000/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "你好，介绍一下你自己",
    "session_id": "sess_demo",
    "memory_id": "default",
    "user_id": "default",
    "dept_tag": "default",
    "kb_id": "default",
    "top_k": 5,
    "use_rag": true,
    "include_long_term_memory": true
  }'
```

## 目录速览

- `main.py`：FastAPI 入口（含静态 Web）
- `rag/`：embedding、Chroma、检索、RAG QA、Agent
- `deepdoc/`：多格式解析（PDF/Word/Excel…）
- `nlp/`：分词/Query 相关（非向量召回主路径）
- `web/static/`：前端页面

## 项目结构

```
ragpulse/
├── main.py                    # FastAPI 入口（挂载 web/static）
├── requirements.txt
├── .env.example
├── readme.md
├── docs/                       # 使用说明/部署/集成
├── api/                        # Agent 等 API 路由
├── web/
│   └── static/                 # 前端（上传入库 / RAG QA / Agent）
├── rag/                        # RAG 主链路
│   ├── ingest/                 # 解析编排、分块、入库（同步/异步）
│   ├── embedding/              # 向量（DashScope text-embedding-v3）
│   ├── retrieval/              # Chroma 持久化、Top‑K 检索、JSON 导出
│   ├── llm/                    # OpenAI 兼容 Chat（百炼 compatible-mode）
│   ├── agent/                  # 会话上下文 + 长期记忆 + 可选 RAG
│   ├── app/                    # 图片描述等小工具
│   └── utils/                  # verbose / 进度等
├── deepdoc/                    # 文档解析（PDF/Word/Excel…）
├── nlp/                        # 分词/Query 相关（非向量召回主路径）
├── memory/                     # 记忆服务（对话消息）
└── common/                     # settings / 存储适配 / 通用工具
```

## 相关文档

- `TEST_RAG.md`：端到端自测与调试说明
- `docs/`：部署/集成/Agent 使用等

# RAGPulse AI Agent 快速入门

## 概述

本指南将帮助您快速搭建并开始使用 RAGPulse AI Agent，体验其上下文记忆、长期记忆和知识库集成的强大功能。

## 前提条件

- Python 3.8 或更高版本
- 已安装必要的 Python 包（通过 `pip install -r requirements.txt`）
- 有效的 API 密钥（Embedding API 和 LLM API）

## 步骤 1: 环境配置

### 1.1 克隆或更新项目

```bash
git clone <repository-url>  # 如果是初次使用
# 或
git pull                    # 如果已存在项目
```

### 1.2 创建虚拟环境并安装依赖

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 1.3 配置 API 密钥

```bash
cp .env.example .env
```

编辑 `.env` 文件，添加您的 API 密钥：

```bash
# 必需：Embedding API 配置
EMBEDDING_API_URL=https://dashscope.aliyuncs.com/api/v1/services/aigc/text-embedding
EMBEDDING_API_KEY=your_actual_api_key_here

# 必需：LLM API 配置
LLM_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
LLM_API_KEY=your_actual_api_key_here
LLM_MODEL=qwen-plus  # 或其他支持的模型

# 可选：服务器配置
HOST=0.0.0.0
PORT=8000
```

## 步骤 2: 启动服务

### 2.1 使用启动脚本（推荐）

```bash
# 给脚本添加执行权限
chmod +x start_agent.sh

# 启动服务
./start_agent.sh
```

或者使用 Python 启动脚本：

```bash
python start_agent.py --host 0.0.0.0 --port 8000
```

服务启动后，您应该能看到类似以下的输出：

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Started reloader process [PID]
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## 步骤 3: 验证服务

### 3.1 检查服务健康状态

在另一个终端窗口中：

```bash
curl http://localhost:8000/health
```

应返回类似：
```json
{
  "status": "ok",
  "missing_hard": [],
  "warnings": [],
  "env_loaded_from": "...",
  "checks": {...}
}
```

### 3.2 查看 API 文档

在浏览器中打开 `http://localhost:8000/docs` 查看所有可用的 API 端点。

## 步骤 4: 使用 AI Agent

### 4.1 与 AI Agent 对话

```bash
curl -X POST "http://localhost:8000/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "你好，介绍一下你自己",
    "session_id": "quick_start_session",
    "memory_id": "quick_start_memory",
    "top_k": 3,
    "use_rag": false,
    "include_long_term_memory": false
  }'
```

### 4.2 启用 RAG 功能（使用知识库）

首先，您需要上传一些文档到知识库：

```bash
curl -X POST "http://localhost:8000/rag/ingest" \
  -F "file=@your_document.pdf" \
  -F "dept_tag=quick_start" \
  -F "kb_id=knowledge_base"
```

然后，启用 RAG 功能进行对话：

```bash
curl -X POST "http://localhost:8000/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "基于已有的文档，解释一下主要概念",
    "session_id": "rag_session",
    "memory_id": "rag_memory",
    "top_k": 5,
    "use_rag": true,
    "include_long_term_memory": false,
    "dept_tag": "quick_start",
    "kb_id": "knowledge_base"
  }'
```

### 4.3 使用长期记忆功能

进行多轮对话，并让 AI 代理记住上下文：

第一轮对话：
```bash
curl -X POST "http://localhost:8000/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "我的名字是张三，我喜欢编程",
    "session_id": "long_term_session",
    "memory_id": "user_profile",
    "top_k": 1,
    "use_rag": false,
    "include_long_term_memory": true
  }'
```

稍后的对话中提及个人信息：
```bash
curl -X POST "http://localhost:8000/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "你还记得我的名字吗？",
    "session_id": "later_session",
    "memory_id": "user_profile",
    "top_k": 1,
    "use_rag": false,
    "include_long_term_memory": true
  }'
```

## 步骤 5: 管理记忆和会话

### 5.1 查看记忆统计

```bash
curl -X GET "http://localhost:8000/agent/memory/list" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "user_profile",
    "user_id": "default"
  }'
```

### 5.2 查看会话列表

```bash
curl -X GET "http://localhost:8000/agent/sessions/list" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "user_profile",
    "user_id": "default"
  }'
```

### 5.3 清除特定记忆

```bash
curl -X POST "http://localhost:8000/agent/memory/clear" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "user_profile",
    "session_id": "long_term_session"
  }'
```

## 高级功能

### 5.1 内容生成

让 AI 代理基于知识库生成内容：

```bash
curl -X POST "http://localhost:8000/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "根据文档内容，写一个关于项目总结的短文",
    "session_id": "writing_session",
    "memory_id": "writing_memory",
    "top_k": 5,
    "use_rag": true,
    "include_long_term_memory": false
  }'
```

### 5.2 多轮复杂对话

AI Agent 能够维持复杂的多轮对话，利用上下文记忆提供连贯的交互体验。

## 故障排除

### 常见问题

1. **API 密钥错误**
   - 检查 `.env` 文件中的 API 密钥是否正确
   - 确认 API 服务是否正常运行

2. **服务启动失败**
   - 确认所需依赖已安装
   - 检查端口是否被其他程序占用

3. **RAG 功能不可用**
   - 确认知识库中已上传文档
   - 检查 dept_tag 和 kb_id 是否匹配

### 检查日志

启用详细日志输出：
```bash
export RAG_VERBOSE=1
./start_agent.sh
```

## 下一步

- 查看 `docs/agent_usage.md` 获取详细的 API 使用说明
- 查看 `docs/deployment.md` 获取生产部署指导
- 探索更多高级功能和定制选项

恭喜！您已经成功启动并开始使用 RAGPulse AI Agent。享受智能对话、记忆管理和知识库集成带来的强大功能吧！
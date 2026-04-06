# RAGPulse AI Agent 部署指南

## 概述

本文档介绍如何部署和运行RAGPulse AI Agent，一个具备上下文记忆、长期记忆和自由互动能力的智能代理系统。

## 系统要求

- Python 3.8 或更高版本
- 至少 4GB RAM (推荐 8GB+)
- 至少 1GB 可用磁盘空间
- 网络连接（用于API调用）

## 快速部署

### 1. 环境准备

```bash
# 克隆项目（如果是新部署）
git clone <repository-url>
cd ragpulse

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制示例配置文件并进行修改：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置以下必要参数：

```bash
# Embedding API配置（必需）
EMBEDDING_API_URL=https://dashscope.aliyuncs.com/api/v1/services/aigc/text-embedding
EMBEDDING_API_KEY=your_dashscope_api_key_here

# LLM API配置（必需）
LLM_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
LLM_API_KEY=your_dashscope_api_key_here
LLM_MODEL=qwen-plus  # 或其他支持的模型名称

# 服务器配置（可选，默认值如下）
HOST=0.0.0.0
PORT=8000

# RAG配置（可选，默认值如下）
RAG_DEPT=default
RAG_KB_ID=default
RAG_USER_ID=default
```

## 启动服务

### 方法1：使用启动脚本（推荐）

```bash
# 给脚本添加执行权限
chmod +x start_agent.sh

# 启动服务
./start_agent.sh
```

### 方法2：使用Python启动脚本

```bash
python start_agent.py --host 0.0.0.0 --port 8000 --workers 1
```

### 方法3：直接使用uvicorn

```bash
# 设置Python路径
export PYTHONPATH=.:$PYTHONPATH

# 启动服务
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

## 验证部署

服务启动后，您可以通过以下方式验证部署是否成功：

1. **健康检查**：
   ```bash
   curl http://localhost:8000/health
   ```

2. **详细健康检查**：
   ```bash
   curl http://localhost:8000/health/detail
   ```

3. **API文档**：
   在浏览器中访问 `http://localhost:8000/docs`

## AI Agent 特性

### 1. 上下文记忆（短期）
- 在单个对话会话中维护对话历史
- 支持多轮对话和上下文感知回复

### 2. 长期记忆
- 在不同会话间持久化存储重要信息
- 基于SQLite的消息存储系统

### 3. 自由互动
- 自然语言对话界面
- 智能意图识别和响应生成

### 4. 知识库集成
- 与现有RAG系统无缝集成
- 支持基于知识库内容的问答和内容生成

## API 使用示例

### 与AI代理对话

```bash
curl -X POST "http://localhost:8000/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "你好，我想了解这个知识库中的信息",
    "session_id": "my_session_123",
    "memory_id": "my_memory",
    "top_k": 5,
    "use_rag": true,
    "include_long_term_memory": true
  }'
```

### 管理记忆

```bash
# 列出记忆信息
curl -X GET "http://localhost:8000/agent/memory/list" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "my_memory",
    "user_id": "default"
  }'

# 清除特定会话的记忆
curl -X POST "http://localhost:8000/agent/memory/clear" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "my_memory",
    "session_id": "my_session_123",
    "user_id": "default"
  }'
```

## 配置选项

### 服务器配置

| 环境变量 | 默认值 | 描述 |
|---------|--------|------|
| HOST | 0.0.0.0 | 服务器监听地址 |
| PORT | 8000 | 服务器监听端口 |
| WORKERS | 1 | 工作进程数 |

### RAG配置

| 环境变量 | 默认值 | 描述 |
|---------|--------|------|
| RAG_DEPT | default | 默认部门标签 |
| RAG_KB_ID | default | 默认知识库ID |
| RAG_USER_ID | default | 默认用户ID |

### API配置

| 环境变量 | 示例值 | 描述 |
|---------|--------|------|
| EMBEDDING_API_URL | https://... | 嵌入模型API地址 |
| EMBEDDING_API_KEY | your_key | 嵌入模型API密钥 |
| LLM_API_URL | https://... | 大语言模型API地址 |
| LLM_API_KEY | your_key | 大语言模型API密钥 |
| LLM_MODEL | qwen-plus | 使用的模型名称 |

## Docker部署（可选）

如果需要使用Docker部署，您可以创建以下Dockerfile：

```Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "start_agent.py", "--host", "0.0.0.0", "--port", "8000"]
```

构建并运行：

```bash
docker build -t ragpulse-agent .
docker run -d -p 8000:8000 -v $(pwd)/.env:/app/.env ragpulse-agent
```

## 故障排除

### 常见问题

1. **API密钥错误**：
   - 检查 `.env` 文件中的API密钥配置
   - 确认API服务是否正常运行

2. **端口被占用**：
   - 更改PORT环境变量或停止占用端口的程序

3. **依赖包缺失**：
   - 确保虚拟环境已激活
   - 重新运行 `pip install -r requirements.txt`

4. **数据库连接问题**：
   - 检查SQLite文件的读写权限
   - 确认 `common/doc_store/sqlite_message_store.py` 配置

### 日志查看

启用详细日志以进行调试：

```bash
export RAG_VERBOSE=1
```

## 性能调优

- 根据服务器性能调整 `WORKERS` 数量
- 定期清理不必要的长期记忆以节省存储空间
- 监控内存使用情况，适时重启服务

## 安全注意事项

- 妥善保管API密钥，不要将其提交到代码仓库
- 限制对服务的网络访问，仅开放必要端口
- 定期轮换API密钥
- 启用HTTPS保护数据传输安全

## 更新与维护

### 更新代码

```bash
git pull origin main
pip install -r requirements.txt
```

### 数据备份

定期备份SQLite数据库文件以防止数据丢失。

## 支持

如遇到问题，请参考：

- [API文档](http://localhost:8000/docs)
- 检查日志输出
- 创建GitHub Issue（如果适用）
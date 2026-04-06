# RAGPulse AI Agent 使用指南

## 功能介绍

RAGPulse AI Agent 是一个具备以下功能的智能代理：

1. **上下文记忆**：在当前对话会话中记住之前的交流内容
2. **长期记忆**：在多个会话之间持久化存储重要信息
3. **自由互动**：支持自然的语言对话
4. **知识库整合**：可以基于已有知识库内容进行回答和创作

## API 接口

### 主要聊天接口
- `POST /agent/chat` - 主要的代理聊天接口，支持上下文记忆、长期记忆和知识库集成

### 记忆管理接口
- `GET /agent/memory/list` - 列出存储的记忆
- `POST /agent/memory/clear` - 清除特定的记忆上下文

### 会话管理接口
- `GET /agent/sessions/list` - 列出对话会话

## 使用示例

### 1. 与AI代理对话

```bash
curl -X POST "http://localhost:8000/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "你好，我想了解这个知识库中的信息",
    "session_id": "session_123",
    "memory_id": "default",
    "top_k": 5,
    "use_rag": true,
    "include_long_term_memory": true
  }'
```

### 2. 列出记忆信息

```bash
curl -X GET "http://localhost:8000/agent/memory/list" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "default",
    "user_id": "default"
  }'
```

### 3. 清除记忆

```bash
curl -X POST "http://localhost:8000/agent/memory/clear" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "default",
    "session_id": "session_123",
    "user_id": "default"
  }'
```

### 4. 列出会话

```bash
curl -X GET "http://localhost:8000/agent/sessions/list" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "default",
    "user_id": "default"
  }'
```

## 参数说明

### Agent Chat 请求参数

- `query` (str): 用户的查询或输入
- `session_id` (str): 当前对话会话的唯一ID
- `memory_id` (str): 记忆上下文的ID (默认: "default")
- `top_k` (int): 从知识库检索的结果数量 (默认: 5)
- `use_rag` (bool): 是否使用RAG知识库检索 (默认: True)
- `include_long_term_memory` (bool): 是否包含长期记忆 (默认: True)
- `user_id` (str): 用户ID (默认: "default")
- `dept_tag` (str): 部门标签 (默认: "default")
- `kb_id` (str): 知识库ID (默认: "default")

## 对话流程

1. **发起对话**：用户发送查询到 `/agent/chat` 接口
2. **上下文获取**：代理获取当前会话的历史对话记录
3. **知识库检索**：如果启用RAG，代理会从知识库中检索相关信息
4. **长期记忆检索**：如果启用长期记忆，代理会检索相关的过往记忆
5. **生成回复**：代理结合所有信息生成自然流畅的回复
6. **记忆存储**：对话内容被保存到短期和长期记忆中

## 配置说明

### 环境变量

确保在 `.env` 文件中正确配置了以下变量：

```bash
# Embedding API配置
EMBEDDING_API_URL=your_embedding_api_url
EMBEDDING_API_KEY=your_embedding_api_key

# LLM API配置
LLM_API_URL=your_llm_api_url
LLM_API_KEY=your_llm_api_key
LLM_MODEL=qwen-plus  # 或其他支持的模型名称

# （可选）多模态模型配置
LLM_VISION_MODEL=qwen-vl-plus
RAG_PUBLIC_BASE_URL=https://your_domain.com  # 公网可访问的基础URL

# 默认RAG参数
RAG_DEPT=default
RAG_KB_ID=default
RAG_USER_ID=default
```

## 应用场景

### 1. 知识库问答
使用AI代理直接查询和解答基于已有知识库的问题。

### 2. 内容创作
利用知识库信息辅助生成文章、报告或其他内容。

### 3. 个性化助手
基于长期记忆提供个性化的服务和支持。

### 4. 持续学习
代理可以通过对话不断积累新的知识和经验。

## 开发者接口

如果您想在自己的应用中集成RAGPulse AI Agent，可以参考以下Python示例：

```python
from rag.agent.service import create_agent

# 创建代理实例
agent = create_agent(
    user_id="user123",
    dept_tag="department_name",
    kb_id="knowledge_base_id"
)

# 进行对话
response = agent.chat(
    query="你想问的问题",
    session_id="unique_session_id",
    memory_id="memory_context_id",
    top_k=5,
    use_rag=True,
    include_long_term_memory=True
)

print(response['response'])
```
#!/bin/bash
# RAGPulse AI Agent 启动脚本

echo "==========================================="
echo "    RAGPulse AI Agent 启动脚本"
echo "==========================================="

# 检查是否已经激活虚拟环境
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "警告: 没有检测到虚拟环境。建议在虚拟环境中运行。"
    read -p "是否继续? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
    fi
else
    echo "检测到已激活的虚拟环境: $VIRTUAL_ENV"
fi

# 检查 .env 文件
ENV_FILE=".env"
if [[ -f "$ENV_FILE" ]]; then
    echo "加载环境配置文件: $ENV_FILE"
    source "$ENV_FILE"
else
    echo "警告: 未找到 $ENV_FILE 文件。请确保已配置必要的环境变量。"
    echo "请参见 .env.example 文件进行配置。"
fi

# 设置必要的环境变量
export PYTHONPATH="${PYTHONPATH}:."

# 启动参数
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-1}

echo
echo "启动参数:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo

# 显示API端点信息
echo "可用API端点:"
echo "  - 健康检查: http://$HOST:$PORT/health"
echo "  - RAG检索: http://$HOST:$PORT/rag/retrieve"
echo "  - RAG问答: http://$HOST:$PORT/rag/qa"
echo "  - AI代理聊天: http://$HOST:$PORT/agent/chat"
echo "  - AI代理记忆列表: http://$HOST:$PORT/agent/memory/list"
echo "  - AI代理会话列表: http://$HOST:$PORT/agent/sessions/list"
echo "  - API文档: http://$HOST:$PORT/docs"
echo

# 启动服务器
echo "正在启动 RAGPulse AI Agent 服务..."
echo "访问 http://$HOST:$PORT 查看服务状态"
echo

# 使用uvicorn启动
uvicorn main:app --host $HOST --port $PORT --workers $WORKERS --reload

# 如果上面的命令失败，尝试使用python直接运行
if [ $? -ne 0 ]; then
    echo
    echo "uvicorn 启动失败，尝试使用 python 直接运行..."
    python main.py
fi
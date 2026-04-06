"""
RAGPulse AI Agent 启动脚本
提供了多种启动选项和配置验证
"""
import os
import sys
import argparse
from pathlib import Path


def check_environment():
    """检查环境配置"""
    print("检查环境配置...")

    # 检查必要的环境变量
    required_vars = ['EMBEDDING_API_KEY', 'LLM_API_KEY', 'LLM_API_URL']
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"⚠️  警告: 以下环境变量未设置: {', '.join(missing_vars)}")
        print("   请参考 .env.example 文件配置环境变量")
    else:
        print("✓ 所需环境变量均已设置")

    # 检查 .env 文件
    env_file = Path(".env")
    if env_file.exists():
        print(f"✓ 找到配置文件: {env_file.absolute()}")
    else:
        print("⚠️  未找到 .env 配置文件，将使用系统环境变量")

    return len(missing_vars) == 0


def start_server(host="0.0.0.0", port=8000, workers=1, reload=False):
    """启动服务器"""
    print(f"\n准备启动服务器...")
    print(f"地址: {host}:{port}")
    print(f"工作进程数: {workers}")
    print(f"热重载: {'开启' if reload else '关闭'}")

    try:
        import uvicorn
        print("\n🚀 启动 RAGPulse AI Agent 服务...")

        # 设置 PYTHONPATH
        os.environ['PYTHONPATH'] = str(Path.cwd()) + ':' + os.environ.get('PYTHONPATH', '')

        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level="info"
        )
    except ImportError:
        print("❌ 未找到 uvicorn，请安装: pip install uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  服务已被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 启动过程中发生错误: {e}")
        sys.exit(1)


def show_endpoints(host="0.0.0.0", port=8000):
    """显示可用的API端点"""
    base_url = f"http://{host}:{port}"
    print("\n🌐 可用的API端点:")
    print(f"  健康检查:           {base_url}/health")
    print(f"  详细健康检查:       {base_url}/health/detail")
    print(f"  RAG检索:            {base_url}/rag/retrieve (POST)")
    print(f"  RAG问答:            {base_url}/rag/qa (POST)")
    print(f"  AI代理聊天:         {base_url}/agent/chat (POST)")
    print(f"  AI代理记忆列表:     {base_url}/agent/memory/list (GET)")
    print(f"  AI代理会话列表:     {base_url}/agent/sessions/list (GET)")
    print(f"  API文档:            {base_url}/docs")
    print(f"  Web界面:            {base_url}/")


def main():
    parser = argparse.ArgumentParser(description="RAGPulse AI Agent 启动器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口 (默认: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数 (默认: 1)")
    parser.add_argument("--reload", action="store_true", help="启用热重载模式")
    parser.add_argument("--check-env", action="store_true", help="仅检查环境配置")

    args = parser.parse_args()

    print("="*50)
    print("    RAGPulse AI Agent 启动器")
    print("="*50)

    # 检查环境
    env_ok = check_environment()

    if args.check_env:
        if env_ok:
            print("\n✅ 环境配置检查通过")
        else:
            print("\n❌ 环境配置存在问题，请检查警告信息")
        return

    if not env_ok:
        print("\n⚠️  环境配置存在问题，但仍将继续启动...")
        input("按 Enter 键继续，或 Ctrl+C 取消...")

    # 显示API端点
    show_endpoints(args.host, args.port)

    # 启动服务器
    start_server(args.host, args.port, args.workers, args.reload)


if __name__ == "__main__":
    main()
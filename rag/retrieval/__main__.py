"""
不启动 HTTP 时自检导出工具：

  PYTHONPATH=. python -m rag.retrieval --help
"""

from __future__ import annotations

import argparse
import os


def _cmd_retrieve(args: argparse.Namespace) -> None:
    from rag.retrieval.json_export import suggest_retrieval_export_path
    from rag.retrieval.rag_retrieval import retrieve_for_query

    export_path = args.export
    if args.auto_export and not export_path:
        export_path = str(suggest_retrieval_export_path(args.query))
    hits = retrieve_for_query(
        query=args.query,
        user_id=os.getenv("RAG_USER_ID", "default"),
        dept_tag=os.getenv("RAG_DEPT", "default"),
        kb_id=os.getenv("RAG_KB_ID", "default"),
        top_k=args.top_k,
        export_path=export_path if (args.export or args.auto_export) else None,
    )
    print(f"hits: {len(hits)}")
    if export_path and (args.export or args.auto_export):
        print(f"exported: {export_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="ragpulse 检索 + JSON 落盘")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("retrieve", help="向量检索，可选导出 JSON")
    r.add_argument("query", type=str, help="查询句")
    r.add_argument("-k", "--top-k", type=int, default=5)
    r.add_argument("-o", "--export", type=str, default=None, help="检索结果 JSON 路径")
    r.add_argument(
        "-a",
        "--auto-export",
        action="store_true",
        help=f"写入默认目录（RAG_EXPORT_DIR 或 data/rag_exports）下自动命名文件",
    )
    r.set_defaults(func=_cmd_retrieve)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

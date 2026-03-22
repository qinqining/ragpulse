# RAG 测试说明（test.py）

本页说明如何**不启动 HTTP 服务**，在本地跑通 **嵌入 → Chroma → 检索 →（可选）LLM 总结**，并查看全链路日志与 JSON 导出。

## 1. 准备

```bash
cd /path/to/ragpulse
pip install -r requirements.txt
cp .env.example .env   # 若还没有
```

在 `.env` 中至少配置（与 DashScope / 百炼一致即可）：

| 变量 | 说明 |
|------|------|
| `EMBEDDING_API_KEY` 或 `LLM_API_KEY` | 通义 API Key，用于 `text-embedding-v3` |
| `EMBEDDING_API_URL` | 可选，默认已是 DashScope 嵌入端点 |
| `LLM_API_URL` / `LLM_API_KEY` | 可选；仅第 5 步「根据检索总结」需要（OpenAI 兼容） |

## 2. PDF 放置

将论文 PDF 放到**仓库根目录**，推荐文件名：

- `attention_is_all_you_need.pdf`  
或  
- `attention_is_all_your_need.pdf`

若根目录**只有一个** `.pdf`，脚本也会自动选用。多个 PDF 时需改名或改 `test.py` 里的 `PDF_CANDIDATES`。

> 说明：`test.py` 使用 **pypdf 抽文本**，适合文字型 PDF。扫描版/大图多需走 **deepdoc + OCR / 视觉**，不在本脚本范围内。

## 3. 全链路日志（print / logging）

- **`[test] ...`**：测试脚本主步骤，**始终输出到 stdout**。
- **`[rag.embed]` / `[rag.chroma]` / `[rag.retrieve]` / `[rag.export]` / `[rag.llm]` / `[rag.picture]`**：RAG 包内诊断信息，**仅当 `RAG_VERBOSE=1`（或 true/yes/on）时**输出到 **stderr**。

推荐：

```bash
export RAG_VERBOSE=1
PYTHONPATH=. python test.py
```

关闭 RAG 内部输出、只看 `[test]`：

```bash
unset RAG_VERBOSE
# 或
export RAG_VERBOSE=0
PYTHONPATH=. python test.py
```

同时会启用 `logging`（`setup_rag_logging()`），级别 DEBUG，与 `RAG_VERBOSE` 联动。

## 4. JSON 导出位置

默认目录：`data/rag_exports/`（可用环境变量 `RAG_EXPORT_DIR` 修改）。

每次运行会生成：

1. **ingest manifest**：入库前每个 chunk 的 `id` / `document` / `metadata` / `embedding_dim`  
   文件名形如：`ingest_ragpulse__...json`
2. **retrieval hits**：检索 query、命中列表、distance 等  
   文件名形如：`retrieve_What_is_...json`

## 5. 可选环境变量（调试）

| 变量 | 默认 | 含义 |
|------|------|------|
| `RAG_TEST_DEPT` | `test_rag` | Chroma collection 的 dept 段 |
| `RAG_TEST_KB` | `attention` | Chroma collection 的 kb 段 |
| `RAG_TEST_QUERY` | 英文 Transformer 问句 | 检索用问题 |
| `RAG_TEST_TOPK` | `5` | 返回条数 |
| `VECTOR_DB_DIR` | `vector_db/chroma` | Chroma 持久化目录 |

脚本启动时会尝试 **删除同名 collection** 再写入，避免重复 `id` 报错。

## 6. 预期现象

1. 打印非空页数、分块数、若干 chunk 预览。  
2. stderr 出现 `[rag.embed]` 逐条嵌入进度。  
3. `[rag.chroma]` 持久化路径、`add` / `query`。  
4. `[rag.export]` 写出 JSON 路径与大小。  
5. `[test]` 打印命中摘要；若配置了 LLM，第 5 步有 `[rag.llm]` 与回答节选。

若嵌入报错，请检查 Key 与网络；若检索 0 条，检查 `RAG_TEST_DEPT` / `RAG_TEST_KB` 是否与入库一致。

### 嵌入中途 `Connection reset by peer` / `Connection aborted`

多为**网络抖动**或**短时间大量新建 HTTPS 连接**被对端断开。已默认使用 **`requests.Session` 连接池**并对 `ConnectionError` / `Timeout` **自动重试**（指数退避）。仍失败时可：

- 在 `.env` 增加 **`EMBEDDING_INTERVAL_SEC=0.15`**（每条嵌入间隔略放慢，减轻限流）；
- 调大 **`EMBEDDING_MAX_RETRIES`**、**`EMBEDDING_RETRY_SLEEP`**（见 `.env.example`）。

## 7. 其它入口

- CLI 仅检索：`PYTHONPATH=. python -m rag.retrieval retrieve "你的问题" -a`  
- HTTP：启动 `uvicorn main:app` 后浏览器打开 **`http://127.0.0.1:8000/`**（极简页面调 `POST /rag/retrieve`）；若用 `test.py` 入过库，页面上 **dept / kb** 保持 `test_rag` / `attention`（或与你的 `RAG_TEST_*` 一致）。

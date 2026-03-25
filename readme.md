# ragpulse

基于 **RAGFlow 设计思路**裁剪的轻量级工程：**嵌入 → 向量库（Chroma）→ 检索 →（可选）LLM 生成**，并保留 **`nlp/`**（Query/分词）、**`deepdoc/`**（多格式解析）、**`memory/`**（对话记忆）等可复用代码。  
目标是在 **去掉 Infinity / ES / 完整 RAGFlow 后端** 的前提下，仍能本地跑通 RAG 与文档解析扩展。

---

## 功能概览（当前已实现）

| 能力 | 说明 |
|------|------|
| **嵌入** | `rag/embedding/qwen_embed.py`：DashScope `text-embedding-v3`（可用 `EMBEDDING_API_KEY` / `LLM_API_KEY`） |
| **向量库** | `rag/retrieval/chroma_client.py`：持久化 Chroma，`collection_name(dept, kb_id)` 做知识库隔离 |
| **检索** | `rag/retrieval/rag_retrieval.py`：`retrieve_for_query`（查询嵌入 + Top-K） |
| **LLM** | `rag/llm/chat_model.py`：OpenAI 兼容 Chat（百炼 compatible-mode 等） |
| **多模态（解析侧）** | `rag/app/picture.py`：页面/插图 → 视觉模型 → 文本描述（供 deepdoc 调用） |
| **记忆存储** | `common/settings.py` + `common/doc_store/sqlite_message_store.py`：`messages.py` 使用的 SQLite 后端 |
| **JSON 检查** | `rag/retrieval/json_export.py`：入库 manifest / 检索结果落盘 |
| **可观测** | `rag/utils/verbose.py`：`RAG_VERBOSE=1` 时 RAG 全链路 `[rag.*]` 输出至 stderr |
| **HTTP 演示** | `main.py`：`GET /`、`GET /health`、`POST /rag/ingest`（同步上传）、`POST /rag/ingest/async` + `GET /rag/ingest/tasks/{task_id}`（异步入库轮询）、`POST /rag/retrieve`（仅检索）、`POST /rag/qa`（检索+LLM 回答） |
| **端到端自测** | 根目录 `test.py` + **`TEST_RAG.md`**（PDF 文本抽取 → 嵌入 → Chroma → 检索） |

---

## 快速开始

### 1. 环境

```bash
cd ragpulse
python -m venv .venv && source .venv/bin/activate   # 可选
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env：至少配置 EMBEDDING_API_KEY（或 LLM_API_KEY）、按需配置 LLM_API_URL / LLM_API_KEY
```

变量说明见 **`.env.example`**。

### 2. 不启动服务：跑 RAG 全链路自测

```bash
export RAG_VERBOSE=1
# 将 PDF 置于项目根目录，例如 attention_is_all_you_need.pdf
PYTHONPATH=. python test.py
```

详见 **`TEST_RAG.md`**。

### 3. 启动 HTTP

```bash
PYTHONPATH=. uvicorn main:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://127.0.0.1:8000/  — 上传入库 + 检索（web/static/index.html）
# 健康检查: GET /health（简版），GET /health/detail（含关键环境变量检查）
# 入库: POST /rag/ingest  multipart: file, dept_tag, kb_id, parser(auto|pdf|pdf_deepdoc|pdf_pypdf|txt|md|docx), pdf_doc_type, ...
# 解析器列表: GET /rag/ingest/options
# 检索: POST /rag/retrieve  JSON: {"query":"...","top_k":5,"dept_tag":"test_rag","kb_id":"attention",...}
# RAG+回答: POST /rag/qa  JSON: 同上 + 可选 "use_vision":true（需 RAG_PUBLIC_BASE_URL + LLM_VISION_MODEL）
# 飞书/微信/App 对接说明: docs/integrations.md
# 长文入库建议走异步: POST /rag/ingest/async -> 返回 task_id，再 GET /rag/ingest/tasks/{task_id} 轮询
```

检索使用的 collection 由 **`RAG_DEPT`、`RAG_KB_ID`**（请求体可覆盖）决定，需与入库时一致（例如先 `test.py` 再在前端填相同 dept/kb）。

#### PDF 入库：deepdoc vs pypdf

- **`pdf` / `auto` + `.pdf`**：优先 **`deepdoc.parser.pdf_parser.RAGFlowPdfParser`**（版面 + OCR/表格等，与 RAGFlow 一致的重依赖）；失败时**自动回退** `pypdf` 按页抽字。
- **`pdf_deepdoc`**：只走 deepdoc，失败即报错（便于排查模型/onnx）。
- **`pdf_pypdf`**：只走文本层，轻量但扫描件效果差。

依赖见 **`requirements.txt`** 中 deepdoc 段（`pdfplumber`、`opencv-python-headless`、`onnxruntime`、`shapely`、`pyclipper`、`xgboost`、`scikit-learn`、`huggingface-hub` 等）；首次运行可能从 HuggingFace 拉取版面/OCR 资源（可设 `HF_ENDPOINT` 镜像）。

**若提示「deepdoc 失败已回退 pypdf」且写明 `无法导入 deepdoc PDF 模块`**：说明 **Python 包未装全**（常见 `No module named 'pdfplumber'` 或 `cv2`）。在项目根执行：

```bash
pip install -r requirements.txt
PYTHONPATH=. python -c "from deepdoc.parser.pdf_parser import RAGFlowPdfParser"
```

第二行会打印**具体缺哪个包**；装齐后应无报错。

**若报错 `cannot import name 'pip_install_torch' from 'common.misc_utils'`**（或其它 `common.*` 缺失）：说明 deepdoc 仍依赖 **RAGFlow 同名 `common` 模块**；本仓库在 `common/misc_utils.py` 中提供 **`pip_install_torch`**、**`thread_pool_exec`** 等与上游对齐的占位/实现，请拉取最新代码。`pip_install_torch` **不会**自动 `pip install torch`，需要 GPU 加速时请自行安装 `torch`。

**若 import 成功但初始化失败**（模型下载、磁盘、`rag/res/deepdoc` 等）：选 **`pdf_pypdf`** 先入库，或检查网络/HF 镜像与 `rag/res/deepdoc` 是否可写。

**若报错 `Failed to load model ... updown_concat_xgb.model` / `binary format ... removed in 3.1`**：deepdoc 自带的 xgb 权重仍是**旧二进制**，与 **XGBoost 3.1+** 不兼容。请 **`pip install "xgboost>=2,<3.1"`**（与 **`requirements.txt`** 中上限一致），或自行用 XGBoost 3.0 按官方文档把模型转为 UBJ/JSON 后替换文件。

**`data/rag_exports` 里两种入库 JSON（与检索无关）**

| 文件名模式 | ``kind`` | 时机 | 内容 |
|------------|----------|------|------|
| ``chunks_pre_embed_*`` | ``ingest_chunks_pre_embed`` | **调用嵌入 API 之前** | 仅解析+分块后的 ``id`` / ``document`` / ``metadata``，**无向量**；顶层含 ``extract_engine``（是否 deepdoc 看此字段） |
| ``ingest_*`` | ``ingest_manifest`` | **已算完 embedding**、写入 Chroma 之前 | 同上 + 每条 ``embedding_dim``（可选 ``embedding_preview``） |

**检索**另会生成 ``retrieve_*``（仅当你在前端勾选导出或 API 传 ``auto_export_retrieval``），与入库无关。

#### PDF 内嵌图与 ``metadata.image_uri``

- **deepdoc / pypdf 文本**：仍只产生正文；另用 **``pypdf`` 按物理页**抽取 PDF **内嵌位图**，保存到 ``web/static/images/<uuid>.png``（与 ``main.py`` 的 ``/static`` 挂载一致）。
- 每个 chunk 的 ``metadata`` 中，若该 **``page``**（与 PDF 页码一致）上有图，则增加 **``image_uri``**：**JSON 字符串**，内容为路径数组，例如 ``["/static/images/abc.png"]``。**不写死域名**，由调用方用当前站点 origin 拼接（浏览器可直接 ``<img src="/static/...">`` 同源访问）。
- 关闭抽图：环境变量 ``RAG_EXTRACT_PDF_IMAGES=false`` 或入库表单 ``extract_pdf_images=false``。
- 检索接口每条 hit 额外带 **``image_uris``**（解析后的列表），便于前端展示。

**入库进度**：默认在运行 uvicorn 的终端打印 ``[ingest] …``（解析 / 嵌入 / Chroma）。关闭：``.env`` 设 ``RAG_INGEST_PROGRESS=0``。更细嵌入日志另设 ``RAG_VERBOSE=1``。

**入库返回 HTTP 503**：`main.py` 将 ``run_ingest`` 中的 ``RuntimeError`` 映射为 503。终端会打印 ``run_ingest RuntimeError → HTTP 503``；浏览器/接口响应体里的 **`detail` 字段**即为具体原因。HF 出现 `Fetching … 100%` 只说明 **模型下载阶段通过**，后面 **DashScope 嵌入**（`EMBEDDING_API_KEY` / 额度）或 **Chroma 写入** 仍可能报错。

### 4. 仅检索 CLI

```bash
PYTHONPATH=. python -m rag.retrieval retrieve "你的问题" -k 5 -a
# -o path.json 指定导出；-a 使用 data/rag_exports 下自动文件名
```

---

## 目录结构（与仓库一致）

```
ragpulse/
├── main.py                 # FastAPI 最小入口
├── test.py                 # RAG 端到端自测（pypdf + Chroma）
├── TEST_RAG.md             # RAG 测试说明
├── requirements.txt
├── .env.example
│
├── common/                 # 公共层（替代 RAGFlow common 中与存储强耦合部分）
│   ├── settings.py         # .env、msgStoreConn（SQLite）
│   ├── constants.py        # MemoryType、LLMType、字段名占位等
│   ├── connection_utils.py # timeout 装饰器
│   ├── string_utils.py     # 字符串工具
│   ├── token_utils.py / file_utils.py / ...
│   └── doc_store/          # DocStoreConnection 占位、sqlite_message_store
│
├── rag/                    # RAG 核心（与文件格式解耦：只处理「文本块 + 向量」）
│   ├── ingest/             # 上传入库：parsers + run_ingest（不做二次切分）
│   │   ├── parsers.py      # auto/pdf/txt/md/docx
│   │   ├── chunking.py
│   │   └── service.py
│   ├── embedding/
│   │   └── qwen_embed.py
│   ├── retrieval/
│   │   ├── chroma_client.py
│   │   ├── rag_retrieval.py
│   │   ├── json_export.py   # ingest / retrieval JSON 落盘
│   │   └── __main__.py      # CLI retrieve
│   ├── llm/
│   │   └── chat_model.py
│   ├── prompts/
│   │   └── generator.py     # 视觉提示词、TOC 占位等
│   ├── app/
│   │   └── picture.py       # 多模态图片描述（OpenAI 兼容 image_url）
│   └── utils/
│       ├── verbose.py       # RAG_VERBOSE、rag_print、setup_rag_logging
│       └── lazy_image.py
│
├── nlp/                    # Query / 分词 / 全文检索逻辑（非向量 RAG）
│   ├── rag_tokenizer.py    # jieba + 可选 OpenCC（无 infinity）
│   ├── query.py / search.py / term_weight.py / synonym.py / surname.py
│   └── __init__.py
│
├── deepdoc/                # 文档解析（PDF/Word/Excel…），按需安装解析依赖
│   ├── parser/             # 延迟导入：避免未装 docx/pdfplumber 时无法 import 包
│   └── vision/             # OCR、版面等
│
├── memory/
│   └── services/
│       └── messages.py     # 记忆 CRUD（依赖 common.settings.msgStoreConn）
│
└── api/                    # 占位：供 deepdoc 视觉链路的 LLMBundle / 租户模型
    └── db/
        ├── services/llm_service.py
        └── joint_services/tenant_model_service.py
```

说明：**规划中的** `web/`、`docker/`、`test/` 分目录、`rag/embedding/__main__.py`、`bge_embed`、`rerank` 等**尚未全部落地**；以仓库实际文件为准。

---

## 架构与数据流

### RAG 主路径（与文件类型无关）

1. **解析（可选）**：`deepdoc` / `docling_parser` / 自写脚本 → 得到文本 chunk + metadata。  
2. **嵌入**：`QwenTextEmbedding.embed_documents` / `embed_query`。  
3. **入库**：`ChromaRagStore.add`（可传 `export_manifest_path` 先写 JSON 再写入 Chroma）。  
4. **检索**：`retrieve_for_query`（可传 `export_path` 写命中 JSON）。  
5. **生成（可选）**：`chat_completions` 拼接检索片段做回答（当前 `test.py` 第 5 步示例）。

**不按扩展名在 `rag/` 内分多套流水线**；按格式选 parser 的编排建议放在**入库脚本或 API 层**，统一调用同一套 `add`。

### 带图的 PDF

- **解析阶段**：可走 deepdoc 视觉页 / `figure_parser` → `rag/app/picture.py` 调多模态 API，**图片在请求体中以 base64 data URL 形式发送**，得到**文字描述**并入 chunk。  
- **向量库**：当前实现以**文本**为主写入 Chroma；**回答阶段**默认 `chat_model` 为纯文本，若要在总结时再喂图需自行扩展。

---

## 模块边界（约定）

| 目录 | 职责 |
|------|------|
| **`rag/`** | 嵌入、向量库、检索编排、LLM 调用、解析用多模态小工具；**不写** Query 解析业务逻辑。 |
| **`nlp/`** | 分词、Query 相关、与 ES/Infinity 风格兼容的检索表达式等（**非** Chroma 向量检索主路径）。 |
| **`deepdoc/`** | 各格式文档 → 结构化文本/块。 |
| **`memory/`** | 对话消息存储；核心逻辑在 `messages.py`，索引名等通过 `common.settings` 适配。 |

---

## JSON 导出与调试

- **环境变量**：`RAG_EXPORT_DIR`（默认 `data/rag_exports`）。  
- **入库前**：`ChromaRagStore.add(..., export_manifest_path=..., export_embedding_preview=...)`。  
- **检索后**：`retrieve_for_query(..., export_path=...)` 或 CLI `-o`/`-a`，或 HTTP `export_path` / `auto_export_retrieval`。  
- **详细说明**：上文「快速开始」+ `TEST_RAG.md`。

---

## 与完整 RAGFlow 的差异（已剥离/替换）

- **无 Infinity 分词依赖**：`nlp/rag_tokenizer.py` 使用 **jieba**（可选 OpenCC）。  
- **无 ES 作为默认文档库**：`memory/utils/es_conn.py` 等仍保留代码，**未接 ragpulse 默认路径**；消息落 **SQLite**（`SqliteMessageStore`）。  
- **`deepdoc/parser/__init__.py`**：**延迟导入**，仅在使用对应解析器时需要 `python-docx`、`pdfplumber` 等。  
- **`api/db/*`**：**最小占位**，满足 `figure_parser` 等对 `LLMBundle` / 租户默认模型的引用；配置来自环境变量。  
- **`common/settings`**：`DOC_ENGINE_INFINITY`、`PARALLEL_DEVICES` 等与 deepdoc 行为对齐。

---

## 规划与扩展（尚未实现或部分实现）

以下来自原产品规划，**当前仓库未完整实现**，按需迭代：

- 高并发：Go 限流 / Redis 任务队列 / Chroma 连接池与批量接口封装等。  
- `rag/retrieval` 与 **memory / RAPTOR / 重排** 的深度融合。  
- 统一 **ingest API**：上传文件 → 按类型选 parser → 嵌入 → `add`。  
- 完整 **`api/apps`**、**`web/demo`**、**`docker/`** 与分模块 **`test/`**。

---

## 许可证说明

仓库内自 RAGFlow / InfiniFlow 复制的文件保留原文件头 **Apache 2.0** 声明；新增与改写文件亦建议遵循相同许可或你方统一许可策略。

---

## 相关文档

- **`TEST_RAG.md`** — RAG 自测、环境变量、`RAG_VERBOSE` 日志说明。  
- **`.env.example`** — 全部环境变量模板。

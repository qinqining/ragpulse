# 与 App / 飞书 / 微信 等对接

ragpulse 本质是 **HTTP API**，在 **你自己的业务服务器**（飞书长连接、微信回调、App 后端）里用任意语言发请求即可，**不必**把飞书 SDK 打进本仓库。

## 整条 RAG + 回答

`POST /rag/qa`（JSON）：

```json
{
  "query": "用户问题",
  "top_k": 5,
  "dept_tag": "test_rag",
  "kb_id": "attention",
  "use_vision": false,
  "vision_max_images": 4
}
```

响应含：`answer`（LLM 文本）、`hits`（检索片段与 `image_uris`）、`used_vision`、`vision_note` 等。

- **仅文本 RAG**：配置 `.env` 中 `LLM_API_URL` / `LLM_API_KEY` / `LLM_MODEL`。
- **多模态（LLM 看图）**：
  1. 入库时勾选 PDF 抽图，使 `metadata.image_uri` 有相对路径；
  2. 设置 **`RAG_PUBLIC_BASE_URL=https://你的公网域名`**（云厂商需能 **GET** 到 `https://域名/static/images/xxx.png`）；
  3. 设置 **`LLM_VISION_MODEL`**（如 `qwen-vl-plus`），请求里 **`use_vision: true`**。

浏览器本地打开 `http://127.0.0.1:8000` 时，**页面仍可**用相对路径 `/static/...` 显示图；**云端 VLM 不行**，必须公网 URL。

## 仅向量检索（不要 LLM）

`POST /rag/retrieve` —— 与上文相同字段但无 `answer`。

## 飞书 / 微信 机器人（思路）

1. 用户在飞书发消息 → 飞书服务器回调 **你的** HTTPS 服务；
2. 你的服务 `POST` 到 ragpulse：`http(s)://ragpulse内网或公网:8000/rag/qa`；
3. 将返回的 `answer`（及可选 `hits`）格式化回飞书/微信消息；若有 `image_uris`，用 **`RAG_PUBLIC_BASE_URL` + 路径** 拼成可点击链接或发图消息。

内网 ragpulse 时，飞书必须能访问到（内网穿透 / 同 VPC / 反向代理）。

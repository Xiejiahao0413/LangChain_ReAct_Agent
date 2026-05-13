# LangChain ReAct Agent - 智能客服与报告生成系统

一个基于 `LangChain + LangGraph + Streamlit + DashScope(Qwen)` 的多工具智能体项目，面向扫地机器人客服场景（也可迁移到其它垂类）。  
项目融合了 ReAct 工具调用、RAG 检索增强、动态提示词切换和流式对话 UI，支持知识库问答与用户使用报告生成。

---

## 目录

- [项目功能](#项目功能)
- [技术栈与依赖](#技术栈与依赖)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [运行方式](#运行方式)
- [核心流程说明](#核心流程说明)
- [日志与数据文件](#日志与数据文件)
- [当前已知问题](#当前已知问题)
- [后续可扩展方向](#后续可扩展方向)

---

## 项目功能

### 1) 多工具 ReAct 智能体

已实现工具能力（`agent/tools/agent_tools.py`）：

- `rag_summarize(query)`：从向量库检索参考资料并总结回答。
- `get_weather(city)`：天气信息示例工具。
- `get_user_location()`：用户城市示例工具。
- `get_user_id()`：用户 ID 示例工具。
- `get_current_month()`：当前月份示例工具。
- `fetch_external_data(action_input)`：读取外部业务数据（CSV），输入格式为 `user_id,month` 或 JSON 字符串。
- `fill_context_for_report()`：触发报告模式上下文标记。

### 2) RAG 检索增强

已实现：

- 基于 `Chroma` 的本地向量库持久化存储（`rag/vector_store.py`）。
- 支持从 `data/` 目录加载 `txt/pdf` 文档。
- 文档切分（`RecursiveCharacterTextSplitter`）与向量化入库。
- 基于文件 `MD5` 去重，避免重复入库（`md5.txt`）。
- `k` 值可配置的检索器召回。

### 3) 动态提示词切换

已实现：

- 普通问答模式提示词。
- 报告生成模式提示词。
- `ReactAgent` 根据用户意图选择普通问答提示词或报告生成提示词，兼容当前 `LangChain 0.1.x` 依赖。

### 4) 流式对话界面

已实现：

- `Streamlit` 聊天界面（`app.py`）。
- 实时字符级流式输出。
- 对话历史会话管理（`st.session_state`）。

### 5) 模块化设计

已实现：

- 模型工厂（`model/factory.py`）统一管理聊天模型与嵌入模型。
- 配置中心（`utils/config_handler.py`）统一加载 YAML。
- 提示词加载器、日志器、文件处理工具解耦。

---

## 技术栈与依赖

### 核心框架

- `LangChain`：Agent、Prompt、Tool、Chain 编排。
- `LangGraph`：运行时与中间件上下文控制。
- `Streamlit`：Web 聊天界面。
- `Chroma`：本地向量数据库。

### 模型与服务

- `ChatTongyi`（阿里云 DashScope）作为聊天模型。
- `DashScopeEmbeddings` 作为向量嵌入模型。

### 其他

- `PyYAML`：配置解析。
- 本地文件系统：知识库、日志、MD5 去重清单、外部 CSV 数据。

---

## 系统架构

用户请求链路如下：

1. 用户在 `Streamlit` 输入问题（`app.py`）。
2. 请求进入 `ReactAgent`（`agent/react_agent.py`）。
3. Agent 根据任务决定是否调用工具（`agent/tools/agent_tools.py`）。
4. 若调用 `rag_summarize`，则进入 RAG 服务：
   - 向量检索（`rag/vector_store.py`）
   - 组装上下文并调用模型生成（`rag/rag_service.py`）
5. Agent 根据用户意图选择普通问答或报告生成执行器。
6. 结果流式返回前端并展示。

---

## 项目结构

```text
LangChain_React_Agent/
├── app.py                         # Streamlit 应用入口
├── README.md
├── requirements.txt               # 标准依赖入口，转发到 requirements.text
├── requirements.text              # 依赖清单
├── md5.txt                        # 已入库文档 MD5 记录
├── agent/
│   ├── react_agent.py             # ReAct Agent 构建与流式执行
│   └── tools/
│       ├── agent_tools.py         # 工具定义（RAG/天气/外部数据等）
│       └── middleware.py          # LangChain 0.1.x 兼容说明与提示词选择辅助函数
├── rag/
│   ├── rag_service.py             # 检索 + 生成总结服务
│   └── vector_store.py            # 文档加载、切分、入库、检索
├── model/
│   └── factory.py                 # 聊天模型与向量模型工厂
├── utils/
│   ├── config_handler.py          # YAML 配置加载
│   ├── file_handler.py            # txt/pdf 加载与文件工具
│   ├── logger_handler.py          # 日志器
│   ├── path_tool.py               # 路径处理
│   └── prompt_loader.py           # 提示词读取
├── config/
│   ├── agent.yml
│   ├── chroma.yml
│   ├── prompts.yml
│   └── rag.yml
├── prompts/
│   ├── main_prompt.txt
│   ├── rag_summarize.txt
│   └── report_prompt.txt
├── data/
│   └── external/
│       └── records.csv            # 外部业务数据示例
└── logs/
```

---

## 快速开始

### 1) 环境要求

- Python 3.10+（推荐）
- Windows / macOS / Linux
- DashScope API Key

### 2) 安装依赖

推荐使用标准依赖入口：

```bash
pip install -r requirements.txt
```

`requirements.text` 仍保留为原始依赖清单，`requirements.txt` 会转发到它。

### 3) 配置环境变量

#### Windows PowerShell

```powershell
$env:DASHSCOPE_API_KEY="your-api-key"
```

#### macOS / Linux

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

### 4) 准备配置与提示词

确认以下文件存在且内容正确：

- `config/rag.yml`
- `config/chroma.yml`
- `config/agent.yml`
- `config/prompts.yml`
- `prompts/main_prompt.txt`
- `prompts/rag_summarize.txt`
- `prompts/report_prompt.txt`

### 5) 启动应用

```bash
streamlit run app.py
```

---

## 配置说明

### `config/rag.yml`

- `chat_model_name`：聊天模型名称（例如 `qwen3-max`）。
- `embedding_model_name`：嵌入模型名称（例如 `text-embedding-v4`）。

### `config/chroma.yml`

- `collection_name`：向量集合名。
- `persist_directory`：向量库持久化目录。
- `k`：检索返回文档数。
- `data_path`：知识库文档目录。
- `md5_hex_store`：MD5 记录文件路径。
- `allow_knowledge_file_type`：允许导入的知识文件类型。
- `chunk_size/chunk_overlap/separators`：文档切分策略。

### `config/agent.yml`

- `external_data_path`：外部业务数据 CSV 路径。

### `config/prompts.yml`

用于配置不同场景的提示词路径。

---

## 运行方式

### 启动 Streamlit UI

```bash
streamlit run app.py
```

### 独立调试模块（可选）

以下文件均内置了 `__main__` 测试入口，可单独运行：

- `agent/react_agent.py`
- `rag/vector_store.py`
- `rag/rag_service.py`
- `utils/config_handler.py`
- `utils/logger_handler.py`
- `utils/path_tool.py`

---

## 核心流程说明

### 知识库入库流程

1. 扫描 `data_path` 中的 `txt/pdf` 文件。
2. 计算文件 MD5，若在 `md5.txt` 存在则跳过。
3. 加载文本并切分为多个文档块。
4. 按批次写入 Chroma 向量库。

### 问答流程

1. Agent 识别用户意图。
2. 需要知识检索时调用 `rag_summarize`。
3. RAG 服务检索并拼接上下文后调用模型生成。
4. 输出以流式方式返回页面。

### 报告流程

1. `ReactAgent` 识别报告类意图。
2. Agent 选择报告生成提示词对应的执行器。
3. Agent 可调用 `get_user_id`、`get_current_month`、`fetch_external_data` 等工具补齐数据。
4. Agent 组合用户信息 + 外部数据工具输出，生成报告内容。

---

## 日志与数据文件

- 日志目录：`logs/`
- 外部数据：`data/external/records.csv`
- 向量库目录：`chroma_db/`
- 文档去重清单：`md5.txt`

---

## FastAPI 与 SSE 接口

项目新增 `api/` 目录，提供可服务化部署的 FastAPI 入口，适合前后端解耦、接口联调和简历项目演示。

启动 API 服务：

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

普通对话接口：

```bash
curl -X POST http://127.0.0.1:8000/api/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"扫地机器人漏扫怎么办？\"}"
```

SSE 流式接口：

```bash
curl -N "http://127.0.0.1:8000/api/chat/stream?query=扫地机器人漏扫怎么办"
```

接口文件：

- `api/app.py`：FastAPI 应用入口，包含 `/health`、`/api/chat`、`/api/chat/stream`
- `api/schemas.py`：请求与响应模型
- `tests/test_api.py`：使用 FastAPI `TestClient` 和假 Agent 覆盖 REST/SSE 行为

---

## Agent 评测模块

项目新增 `evaluation/` 目录，用于评估 ReAct Agent 在典型任务中的工具调用与回答质量。

- `evaluation/agent_eval_samples.jsonl`：40 条评测样本，覆盖知识库问答、天气工具、报告生成、RAG+业务数据混合任务、通用闲聊和边界问题。
- `evaluation/agent_evaluator.py`：评测脚本，会读取 Agent 执行轨迹中的 `intermediate_steps`，统计工具调用正确率、任务完成率、知识库引用率等指标。
- `evaluation/README.md`：评测集字段、运行方式和指标定义说明。

只校验样本集：

```bash
python -m evaluation.agent_evaluator --dry-run
```

运行部分真实评测：

```bash
python -m evaluation.agent_evaluator --limit 5
```

完整评测结果会写入 `evaluation/results/`，该目录属于本地运行产物，已加入 `.gitignore`。

---

## 已修复问题

以下问题已在当前版本中处理：

1. `config/prompts.yml` 键名已与 `utils/prompt_loader.py` 对齐。
2. `app.py` 已保存完整流式回复。
3. `fetch_external_data` 已稳定返回字符串，并使用 `csv` 解析 CSV。
4. 已新增 `requirements.txt` 标准入口。
5. 已补充 `.gitignore` 和 `tests/smoke_test.py`。

---

## 后续可扩展方向

- 接入真实天气/用户画像接口替换示例工具。
- 将随机用户信息改为会话态或登录态真实信息。
- 增加自动化测试（单元测试、集成测试、回归测试）。
- 引入多轮记忆与长期记忆策略。
- 增强报告生成模板（可视化指标、结构化输出）。

---

## 许可证

当前仓库未声明明确 License。若对外发布，建议补充 `LICENSE` 文件（如 MIT/Apache-2.0）。

# 大悟智能体 (Dawu Agent)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey" alt="Platform">
</p>

**大悟智能体**是一款面向数据分析与报告生成场景的企业级 AI Agent 平台。它采用 Harness Engineering 模式构建，内置多模型协作、向量记忆、安全沙箱、搜索引擎集成与可扩展工具生态，能够自主完成从数据读取、清洗、分析到可视化报告输出的完整工作流。

> 本项目在开发过程中使用了 [agent-harness-engineer](https://github.com/sofild/agent-harness-engineer) Skill 进行架构设计与代码生成指导。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| **多模型统一调度** | 兼容 OpenAI、Anthropic Claude、通义千问、DeepSeek 等主流模型，支持按场景自动路由 |
| **数据分析工具链** | 内置 pandas/SQL 查询、数据可视化、报告生成（Markdown/HTML/Excel） |
| **智能搜索增强** | 集成 Bing、百度千帆、Tavily 三大搜索引擎，支持时间感知自动过滤 |
| **向量长期记忆** | 基于 ChromaDB 的语义记忆，自动关联历史会话上下文 |
| **多 Agent 协作** | 数据工程师、分析师、报告撰写员等角色自动分工协作 |
| **企业级安全** | 6 层安全防御（权限管理、Hook 校验、沙箱隔离、审计日志） |
| **MCP 协议扩展** | 支持 Model Context Protocol，可快速接入外部工具与数据源 |

---

## 快速开始

### 环境要求

- Python 3.10+
- Windows / macOS / Linux
- 无需 Docker（沙箱自动降级为路径隔离）

### 安装

```bash
# 克隆仓库
git clone <repo-url>
cd dawu-agent

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖（国内用户推荐清华镜像）
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **注意**：Windows 用户若遇到 `onnxruntime` 错误，请执行：
> ```bash
> pip install chromadb==0.4.24 "numpy<2.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 配置

```bash
# 复制环境模板
cp .env.example .env  # Windows: copy .env.example .env
```

编辑 `.env`，填写模型 API 密钥：

```bash
MODEL_API_KEY=your-api-key
MODEL_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
MODEL_API_TYPE=openai

# 搜索引擎 API Key（可选）
BAIDU_API_KEY=your-baidu-key
TAVILY_API_KEY=your-tavily-key
```

### 运行

```bash
# 交互式 CLI（推荐）
python -m dawu_agent.cli.main run

# 启动 API 服务
python -m dawu_agent.cli.main server
```

---

## 系统架构

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────┐
│  Agent 核心（状态机 + 7 个恢复点）            │
│  • 流式事件输出                              │
│  • 自动上下文压缩                            │
│  • 记忆注入与检索                            │
└──────────┬────────────────┬─────────────────┘
           │                │
    ┌──────┴──────┐  ┌──────┴──────┐
    ▼             ▼  ▼             ▼
┌──────┐   ┌──────────┐   ┌──────────┐
│  LLM │   │ 工具注册表 │   │ 向量记忆  │
│ 路由 │   │ + MCP    │   │ ChromaDB │
└──────┘   └──────────┘   └──────────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌──────┐  ┌────────┐  ┌──────────┐
│文件操作│  │数据分析│  │搜索引擎  │
│沙箱执行│  │可视化  │  │Bing/Baidu│
└──────┘  └────────┘  └──────────┘
```

---

## 配置体系（7 级优先级）

| 优先级 | 来源 | 说明 |
|:------:|------|------|
| 1 | 命令行参数 | `--env production` |
| 2 | 特性开关 | `ENABLE_MULTI_AGENT=true` |
| 3 | 策略规则 | 管理员配置 |
| 4 | 远程配置 | etcd / Consul |
| 5 | 本地覆盖 | `config/local.yaml` |
| 6 | 环境配置 | `config/development.yaml` |
| 7 | 用户全局 | `~/.dawu/settings.yaml` |

---

## 项目结构

```
dawu-agent/
├── src/dawu_agent/          # 核心源码
│   ├── core/                # Agent 主循环、状态机、事件流
│   ├── llm/                 # 多厂商 LLM 适配与路由
│   ├── tools/               # 内置工具 + MCP 适配
│   ├── context/             # 记忆管理、上下文压缩
│   ├── security/            # 权限、Hook、沙箱、审计
│   ├── multi/               # 多 Agent 协调器
│   ├── config/              # 7 级配置加载器
│   └── observability/       # 日志、指标、链路追踪
├── config/                  # 环境配置文件
├── sessions/                # 会话 WAL 持久化
├── memory/                  # 向量数据库文件
├── workspace/               # 沙箱工作目录
├── skills/                  # 可扩展 Skill 目录
└── tests/                   # 测试用例
```

---

## 开发

```bash
# 运行测试
pytest

# 代码检查
ruff check src

# 类型检查
mypy src

# 格式化
ruff format src
```

---

## 常见问题

**Q: Windows 上提示 `onnxruntime` 未安装？**
A: ChromaDB 的已知兼容性问题。已降级到 `chromadb==0.4.24`，如仍有问题请确保使用 Python 3.10 和 `numpy<2.0`。

**Q: 如何添加自定义工具？**
A: 在 `skills/` 目录下创建新模块，实现 `@skill` 装饰器标记的函数，Agent 会自动发现注册。

**Q: 沙箱隔离级别如何切换？**
A: 编辑 `config/development.yaml`：
```yaml
sandbox:
  isolation_level: path   # 无 Docker 时使用 path
```

---

## 开发背景

本项目在开发过程中使用了 [agent-harness-engineer](https://github.com/sofild/agent-harness-engineer) Skill 进行架构设计与代码生成指导。该 Skill 提供了生产级 Agent 系统的构建范式，帮助团队快速搭建起包含状态机、Hook 系统、多模型路由、安全沙箱等核心模块的完整框架。

---

## 许可证

[Apache 2.0](LICENSE)

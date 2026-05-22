# Dawu Agent (大悟智能体)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey" alt="Platform">
</p>

**Dawu Agent** is an enterprise-grade AI Agent platform designed for data analysis and automated report generation. Built with the Harness Engineering pattern, it features multi-model orchestration, vector memory, secure sandboxing, search engine integration, and an extensible tool ecosystem — enabling autonomous end-to-end workflows from data ingestion to visualized report output.

> This project was architected and developed with guidance from the [agent-harness-engineer](https://github.com/sofild/agent-harness-engineer) Skill.

---

## Key Capabilities

| Capability | Description |
|------------|-------------|
| **Multi-Model Orchestration** | Unified interface for OpenAI, Anthropic Claude, Qwen, DeepSeek, and more. Scenario-based automatic routing. |
| **Data Analysis Toolchain** | Built-in pandas/SQL query, data visualization, and report generation (Markdown/HTML/Excel). |
| **Intelligent Search** | Integrated Bing, Baidu Qianfan, and Tavily search with time-aware auto-filtering. |
| **Vector Long-Term Memory** | ChromaDB-based semantic memory with automatic historical context retrieval. |
| **Multi-Agent Collaboration** | Hierarchical agent teams (Data Engineer, Analyst, Report Writer) with automatic task delegation. |
| **Enterprise Security** | 6-layer defense: permission management, Hook validation, sandbox isolation, and audit logging. |
| **MCP Protocol Extension** | Model Context Protocol support for rapid integration of external tools and data sources. |

---

## Quick Start

### Requirements

- Python 3.10+
- Windows / macOS / Linux
- No Docker required (sandbox auto-downgrades to path isolation)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd dawu-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

> **Note**: Windows users encountering `onnxruntime` errors should run:
> ```bash
> pip install chromadb==0.4.24 "numpy<2.0"
> ```

### Configuration

```bash
# Copy environment template
cp .env.example .env  # Windows: copy .env.example .env
```

Edit `.env` and fill in your model API credentials:

```bash
MODEL_API_KEY=your-api-key
MODEL_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
MODEL_API_TYPE=openai

# Search engine API keys (optional)
BAIDU_API_KEY=your-baidu-key
TAVILY_API_KEY=your-tavily-key
```

### Run

```bash
# Interactive CLI (recommended)
python -m dawu_agent.cli.main run

# Start API server
python -m dawu_agent.cli.main server
```

---

## System Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────┐
│  Agent Core (State Machine + 7 Resume Sites) │
│  • Streaming event output                    │
│  • Automatic context compression             │
│  • Memory injection & retrieval              │
└──────────┬────────────────┬─────────────────┘
           │                │
    ┌──────┴──────┐  ┌──────┴──────┐
    ▼             ▼  ▼             ▼
┌──────┐   ┌──────────┐   ┌──────────┐
│ LLM  │   │ Tool     │   │ Vector   │
│Router│   │Registry  │   │ Memory   │
│      │   │+ MCP     │   │ ChromaDB │
└──────┘   └──────────┘   └──────────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌──────┐  ┌────────┐  ┌──────────┐
│ File │  │ Data   │  │ Search   │
│ Ops  │  │ Analysis│  │ Engines  │
│Sandbox│  │Viz     │  │Bing/Baidu│
└──────┘  └────────┘  └──────────┘
```

---

## Configuration Hierarchy (7-Level Priority)

| Priority | Source | Example |
|:--------:|--------|---------|
| 1 | CLI Arguments | `--env production` |
| 2 | Feature Flags | `ENABLE_MULTI_AGENT=true` |
| 3 | Policy Rules | Admin-configured |
| 4 | Remote Config | etcd / Consul |
| 5 | Local Overrides | `config/local.yaml` |
| 6 | Environment | `config/development.yaml` |
| 7 | User Global | `~/.dawu/settings.yaml` |

---

## Project Structure

```
dawu-agent/
├── src/dawu_agent/          # Core source code
│   ├── core/                # Agent main loop, state machine, event stream
│   ├── llm/                 # Multi-vendor LLM adapter & routing
│   ├── tools/               # Built-in tools + MCP adapter
│   ├── context/             # Memory management, context compression
│   ├── security/            # Permissions, Hooks, sandbox, audit
│   ├── multi/               # Multi-agent coordinator
│   ├── config/              # 7-level configuration loader
│   └── observability/       # Logging, metrics, distributed tracing
├── config/                  # Environment configuration files
├── sessions/                # Session WAL persistence
├── memory/                  # Vector database files
├── workspace/               # Sandbox working directory
├── skills/                  # Extensible Skill directory
└── tests/                   # Test suite
```

---

## Development

```bash
# Run tests
pytest

# Linting
ruff check src

# Type checking
mypy src

# Formatting
ruff format src
```

---

## FAQ

**Q: `onnxruntime` not found on Windows?**
A: Known ChromaDB compatibility issue. Downgraded to `chromadb==0.4.24`. If issues persist, ensure Python 3.10 and `numpy<2.0`.

**Q: How to add custom tools?**
A: Create a new module in `skills/` with `@skill` decorated functions. The Agent auto-discovers and registers them.

**Q: How to switch sandbox isolation level?**
A: Edit `config/development.yaml`:
```yaml
sandbox:
  isolation_level: path   # Use 'path' when Docker is unavailable
```

---

## Development Background

This project was architected and developed with guidance from the [agent-harness-engineer](https://github.com/sofild/agent-harness-engineer) Skill, which provides production-grade Agent system building patterns. It helped the team rapidly establish a complete framework including state machines, Hook systems, multi-model routing, and secure sandboxing.

---

## License

[MIT](LICENSE)

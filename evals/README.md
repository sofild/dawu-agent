# Phase 8 Evaluation Suite

Agent 可靠性评估套件。目标不是"看起来能用"，而是**用确定性检查证明 Agent 可靠**。

## 文件结构

```
evals/
├── __init__.py        # 包标记
├── scenarios.py       # 评估场景定义（11 个场景，4 个类别）
├── graders.py         # 4 个确定性 grader 函数
├── baseline.json      # 基线通过率/分数（初始全 0）
└── README.md          # 本文件
```

## 如何运行评估

### 基本用法

```python
import asyncio
from evals.scenarios import ALL_SCENARIOS
from evals.graders import grade_all
from dawu_agent.config.loader import Settings
from dawu_agent.observability.telemetry import TelemetryManager
from dawu_agent.core.agent import Agent

async def run_evals():
    settings = Settings()
    telemetry = TelemetryManager(settings)
    telemetry.initialize()
    agent = Agent(settings=settings, telemetry=telemetry)
    await agent.initialize()

    results = {}
    for scenario in ALL_SCENARIOS:
        events = []
        async for event in agent.run_stream(scenario.user_input):
            events.append(event)
        results[scenario.id] = grade_all(scenario, events)

    return results

asyncio.run(run_evals())
```

### pass^k 概念

**pass^k**（pass-at-k）：对同一场景运行 k 次，**只有 k 次全部通过才算通过**。

这比单次通过严格得多——它证明 Agent 不是偶尔能用，而是**每次都能用**。

```python
k = 3
for scenario in ALL_SCENARIOS:
    all_passed = True
    for _ in range(k):
        events = await run_agent(scenario.user_input)
        grades = grade_all(scenario, events)
        if not all(g["passed"] for g in grades.values()):
            all_passed = False
            break
    scenario_passed = all_passed  # pass^k
```

推荐 k=3 用于 CI，k=5 用于发布前回归测试。

## 四个 Grader 检查什么

| Grader 函数 | 检查内容 | 通过阈值 |
|-------------|---------|---------|
| `grade_task_completion` | 是否产出了非空的 `FinalResponseEvent`（任务完成） | score == 1.0 |
| `grade_tool_selection` | 期望的工具是否被调用；禁止的工具是否被避开 | score == 1.0 |
| `grade_trajectory_quality` | 轮次不超限；出错后是否产出 FinalResponse；状态不是意外 `error`；无连续错误循环 | score >= 0.7 |
| `grade_cost` | 基于轮次/工具调用/错误估算成本是否在预算内 | score >= 0.5 |

每个 grader 返回 `{"passed": bool, "score": float, "detail": str}`：
- `passed`: 是否通过
- `score`: 0.0 – 1.0 的数值分数
- `detail`: 人类可读的检查结果说明

## 场景覆盖

| ID | 类别 | 说明 |
|----|------|------|
| ts-001 | tool_selection | 读取本地文件 → file_read |
| ts-002 | tool_selection | pandas 数据分析 → data_query |
| ts-003 | tool_selection | 列出目录文件 → file_list |
| skill-001 | tool_selection | skill trigger 关键词抑制 web_search |
| err-001 | error_recovery | 读取不存在的文件 → 优雅报错 |
| err-002 | error_recovery | 模型不可用时重试恢复 |
| err-003 | error_recovery | 空文件/空回复恢复 |
| err-004 | error_recovery | 非可重试错误优雅终止 |
| ctx-001 | context_management | 长对话多文件读取触发压缩 |
| ctx-002 | context_management | 多轮"继续"续接 |
| budget-001 | general | token/turn 预算限制执行 |

## CI 门禁

**通过条件**（两者必须同时满足）：

1. **pass_rate >= 80%**：至少 80% 的场景所有 grader 全部通过
2. **无回归**：当前 pass_rate 不低于 `baseline.json` 中记录的 pass_rate

```python
def ci_gate(results, baseline):
    total = len(results)
    passed = sum(
        1 for r in results.values()
        if all(g["passed"] for g in r.values())
    )
    pass_rate = passed / total

    no_regression = pass_rate >= baseline["pass_rate"]
    meets_threshold = pass_rate >= 0.80

    return meets_threshold and no_regression
```

### 更新基线

当 Agent 改进后，更新 `baseline.json` 作为新基线：

```python
import json

def update_baseline(results):
    total = len(results)
    passed = sum(
        1 for r in results.values()
        if all(g["passed"] for g in r.values())
    )
    pass_rate = passed / total
    avg_score = sum(
        sum(g["score"] for g in r.values()) / len(r)
        for r in results.values()
    ) / total

    baseline = {
        "total_scenarios": total,
        "pass_rate": pass_rate,
        "avg_score": avg_score,
        "categories": {},  # 按类别计算
        "last_updated": "2026-09-19",
    }
    with open("evals/baseline.json", "w") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
```

**仅当新 pass_rate 高于旧基线时才更新**，防止偶发通过拉低标准。

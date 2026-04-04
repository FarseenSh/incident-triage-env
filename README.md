---
title: Incident Triage Environment Server
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Incident Response Triage Environment

> **Train AI agents to diagnose and fix production outages like expert SREs.**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Tasks](https://img.shields.io/badge/Tasks-5-green)]()
[![License](https://img.shields.io/badge/License-BSD--3--Clause-orange)]()

An OpenEnv-compatible RL environment where AI agents investigate alerts, trace dependency chains, identify root causes, and remediate simulated production incidents across an 8-service microservice architecture.

---

## Why This Matters

Every tech company has on-call SREs. When production breaks at 3 AM, an engineer must:

1. **Triage** a wall of firing alerts (most are symptoms, not causes)
2. **Investigate** logs and metrics across interconnected services
3. **Trace** failures through dependency chains to find the root cause
4. **Remediate** with the correct action on the correct service

This environment captures that exact reasoning challenge — with red herrings, cascading failures, and intermittent issues that genuinely fool frontier LLMs.

---

## Quick Start

```bash
# Install
pip install openenv-core fastmcp uvicorn fastapi pydantic openai

# Start the server
uvicorn incident_triage_env.server.app:app --host 0.0.0.0 --port 8000
```

```python
# Connect and run an episode
import asyncio
from incident_triage_env import IncidentTriageEnv
from openenv.core.env_server.mcp_types import CallToolAction

async def main():
    async with IncidentTriageEnv(base_url="http://localhost:8000") as env:
        await env.reset(task_name="easy_oom_crash")

        # Investigate
        await env.call_tool("get_alerts")
        await env.call_tool("read_logs", service="order-service", lines=50)
        await env.call_tool("check_metrics", service="order-service", metric="all")

        # Diagnose & fix
        await env.call_tool("set_severity", level="P2")
        await env.call_tool("diagnose",
            root_cause_service="order-service",
            root_cause_category="memory_exhaustion")
        await env.call_tool("remediate",
            action="restart_service", target_service="order-service")

        # Submit (ends episode, triggers grading)
        result = await env.step(CallToolAction(tool_name="submit_report", arguments={}))
        print(f"Reward: {result.observation.reward}")  # 1.0 for perfect diagnosis

asyncio.run(main())
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Agent (LLM)                          │
│         Investigates → Diagnoses → Remediates           │
└──────────────────────┬──────────────────────────────────┘
                       │ MCP Tool Calls
┌──────────────────────▼──────────────────────────────────┐
│              Incident Triage Environment                │
│                                                         │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐ │
│  │ Alerts  │  │   Logs   │  │ Metrics │  │ Topology │ │
│  └────┬────┘  └────┬─────┘  └────┬────┘  └────┬─────┘ │
│       └─────────┬──┴────────┬────┘             │       │
│           ┌─────▼───────────▼──────────────────▼──┐    │
│           │         Simulation Layer              │    │
│           │  8 services, dependency graph,        │    │
│           │  scenario-specific fault injection    │    │
│           └───────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**8 interconnected services:**
`api-gateway` → `auth-service`, `order-service` → `inventory-service`, `payment-service`, `database`, `cache` → `message-queue`

---

## Available Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `get_alerts` | — | Get all firing alerts with incident briefing |
| `read_logs` | `service`, `lines` (default 50) | Read timestamped log lines from a service |
| `check_metrics` | `service`, `metric` (default "all") | Check cpu/memory/latency/error_rate/connections |
| `get_service_topology` | — | View the microservice dependency graph |
| `set_severity` | `level` (P1/P2/P3/P4) | Classify incident severity |
| `diagnose` | `root_cause_service`, `root_cause_category` | Submit root cause diagnosis |
| `remediate` | `action`, `target_service` | Apply a fix to a service |
| `submit_report` | — | Finalize report — ends episode, triggers grading |

<details>
<summary><b>Valid enum values</b></summary>

**Root cause categories:** `memory_exhaustion`, `connection_pool_exhaustion`, `clock_skew`, `disk_full`, `cpu_throttling`, `network_partition`, `config_error`, `dependency_failure`

**Remediation actions:** `restart_service`, `scale_up`, `rollback_deploy`, `config_change`, `failover`, `clear_cache`, `increase_pool_size`

</details>

---

## 5 Incident Scenarios

| # | Task | Difficulty | What Happens | The Trap |
|---|------|-----------|--------------|----------|
| 1 | `easy_oom_crash` | 🟢 Easy | order-service OOM crash | None — alerts and logs clearly point to memory exhaustion |
| 2 | `medium_cascade` | 🟡 Medium | Database pool exhaustion cascades through 4 services | CRITICAL alerts fire on api-gateway (symptom). Database only shows WARN |
| 3 | `medium_disk_full` | 🟡 Medium | Database disk full — writes fail, reads work | Asymmetric failure. Order-service and payment-service look broken but root cause is storage |
| 4 | `hard_intermittent` | 🔴 Hard | auth-service clock skew + CPU red herring on order-service | 8 alerts. order-service has CRITICAL CPU alert with ERROR logs (it's just a cron job). Auth clock skew buried in 8% of logs |
| 5 | `hard_network_partition` | 🔴 Hard | Network partition isolates payment-service from message-queue | 8 alerts. DB vacuum (red herring), auth token spike (red herring). Must identify network_partition and apply failover |

---

## Reward Function

### Terminal Reward (4 components, max 1.0)

| Component | Weight | Exact Match | Partial Credit |
|-----------|--------|-------------|----------------|
| Severity | 0.20 | Correct P-level | Off-by-one = 0.10 |
| Service ID | 0.20 | Root cause service | Any affected service = 0.10 |
| Root Cause | 0.30 | Correct category | — |
| Remediation | 0.30 | Correct action + target | Correct action only = 0.10 |

### Per-Step Signals

| Action | Reward | Condition |
|--------|--------|-----------|
| Investigate relevant service | +0.02 | `read_logs`/`check_metrics` on affected service |
| First `get_service_topology` | +0.01 | One-time bonus |
| First `get_alerts` | +0.01 | One-time bonus |
| Restart healthy service | -0.03 | Penalty for destructive action |
| Max steps exhausted (20) | -0.10 | Episode forced termination |

**Total** = terminal + investigation + penalty, clamped to [0.0, 1.0]

---

## Baseline Scores — Multi-Model Benchmark

We benchmarked three models across all five tasks. The environment produces consistent difficulty gradients and genuinely differentiates model capabilities.

| Model | Easy | Med Cascade | Med Disk | Hard Intermit. | Hard NetPart | Avg |
|-------|------|-------------|----------|----------------|--------------|-----|
| **Qwen 3.6 Plus** | 1.00 | 1.00 | 0.68 | 0.75 | 0.82 | **0.85** |
| **Kimi K2.5** | 1.00 | 1.00 | 1.00 | 0.00 | 0.80 | **0.76** |
| **Gemma 4 26B** | 0.82 | 0.76 | 0.90 | 0.66 | 0.55 | **0.74** |

**Key findings:**
- All models ace the easy task but struggle on hard scenarios (0.00–0.82)
- Different models fail in different ways — Kimi completely fails `hard_intermittent` (gives up without submitting), Gemma gets wrong root cause on `hard_network_partition`
- The red herring mechanics genuinely fool frontier models
- Score variance across tasks proves the grader differentiates behavior, not just model identity

### Reproduce

```bash
# Free baseline (HF Router + Qwen 2.5 72B)
HF_TOKEN=hf_xxx python inference.py --base-url http://localhost:8000

# Multi-model benchmark
ANTHROPIC_API_KEY=x MOONSHOT_API_KEY=y HF_TOKEN=z python inference.py --run-all
```

---

## Setup

### Docker

```bash
docker build -t incident-triage-env .
docker run -p 8000:8000 incident-triage-env
```

### Local

```bash
pip install -e .
uvicorn incident_triage_env.server.app:app --host 0.0.0.0 --port 8000
```

### Deploy to HF Spaces

```bash
pip install openenv-core
openenv push --repo-id YOUR_USERNAME/incident-triage-env
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | For inference | LLM endpoint (default: HF Router) |
| `MODEL_NAME` | For inference | Model ID (default: Qwen/Qwen2.5-72B-Instruct) |
| `HF_TOKEN` | For inference | Hugging Face API token |
| `ANTHROPIC_API_KEY` | Optional | For Claude Sonnet baseline |
| `MOONSHOT_API_KEY` | Optional | For Kimi K2.5 baseline |

---

## RL Training Integration

Compatible with TRL GRPO for reinforcement learning from environment feedback:

```python
from trl import GRPOTrainer, GRPOConfig
from incident_triage_env.server.incident_environment import IncidentTriageEnvironment
from openenv.core.env_server.mcp_types import CallToolAction

def reward_fn(completions, task_name="easy_oom_crash"):
    rewards = []
    for completion in completions:
        env = IncidentTriageEnvironment()
        env.reset(task_name=task_name)
        for tool_call in parse_tool_calls(completion):
            obs = env.step(CallToolAction(
                tool_name=tool_call["name"],
                arguments=tool_call["args"]))
            if obs.done:
                rewards.append(obs.reward)
                break
        else:
            rewards.append(0.0)
    return rewards

trainer = GRPOTrainer(
    model=model,
    config=GRPOConfig(output_dir="./grpo-triage", num_generations=4),
    reward_funcs=[reward_fn],
    train_dataset=dataset,
)
trainer.train()
```

---

## Project Structure

```
incident_triage_env/
├── __init__.py                     # Package exports
├── models.py                       # IncidentTriageState + enums
├── client.py                       # MCPToolClient (pass-through)
├── openenv.yaml                    # OpenEnv manifest
├── pyproject.toml                  # Dependencies
├── Dockerfile                      # Multi-stage Docker build
├── inference.py                    # Baseline agent (multi-model)
├── server/
│   ├── app.py                      # FastAPI app factory
│   └── incident_environment.py     # Core MCPEnvironment subclass
├── scenarios/
│   ├── base.py                     # ScenarioBase ABC + GroundTruth + Registry
│   ├── easy_oom_crash.py           # Task 1: Single service OOM
│   ├── medium_cascade.py           # Task 2: Cascading dependency failure
│   ├── medium_disk_full.py         # Task 3: Database disk full
│   ├── hard_intermittent.py        # Task 4: Auth clock skew + red herrings
│   └── hard_network_partition.py   # Task 5: Network partition + red herrings
└── simulation/
    ├── service_graph.py            # 8-service microservice topology
    ├── log_generator.py            # Realistic timestamped log generation
    ├── metrics_store.py            # Per-service metrics with overrides
    └── alert_engine.py             # Alert creation and severity sorting
```

---

*Built for the [Meta x PyTorch x HuggingFace OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv)*

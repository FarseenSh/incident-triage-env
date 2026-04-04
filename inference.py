"""
Inference Script — Incident Response Triage Environment
========================================================
MANDATORY:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=incident_triage_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Usage:
    python inference.py
    python inference.py --base-url https://your-space.hf.space
"""
import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openenv.core.env_server.mcp_types import CallToolAction

# Try local import first, fall back to installed package
try:
    from client import IncidentTriageEnv
except ImportError:
    from incident_triage_env.client import IncidentTriageEnv

# ─── Configuration ────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

MAX_STEPS = 20
BENCHMARK = "incident_triage_env"
TASKS = ["easy_oom_crash", "medium_cascade", "medium_disk_full", "hard_intermittent", "hard_network_partition"]

SYSTEM_PROMPT = """You are an expert SRE (Site Reliability Engineer) investigating a production incident.
You have access to these tools:
- get_alerts(): View all currently firing alerts
- read_logs(service, lines): Read log lines from a service
- check_metrics(service, metric): Check metrics (cpu/memory/latency/error_rate/connections/all)
- get_service_topology(): View the microservice dependency graph
- set_severity(level): Classify severity (P1/P2/P3/P4)
- diagnose(root_cause_service, root_cause_category): Submit root cause diagnosis
- remediate(action, target_service): Apply a fix
- submit_report(): Finalize your incident report (ends the episode)

Investigation strategy:
1. First check alerts to understand what's firing
2. Read logs from the most-alerted services
3. Check metrics to confirm hypotheses
4. Trace the dependency chain to find the ROOT CAUSE (not just symptoms)
5. Set severity, diagnose, remediate, then submit_report

IMPORTANT: Always set severity, diagnose, AND remediate BEFORE calling submit_report."""


# ─── Structured stdout logging ────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ─── Tool conversion ─────────────────────────────────────

def tools_to_openai_format(mcp_tools) -> List[dict]:
    """Convert MCP tool definitions to OpenAI function-calling format."""
    result = []
    for tool in mcp_tools:
        props = {}
        required = []
        if tool.input_schema and "properties" in tool.input_schema:
            for name, schema in tool.input_schema["properties"].items():
                props[name] = {
                    "type": schema.get("type", "string"),
                    "description": schema.get("description", ""),
                }
            required = tool.input_schema.get("required", [])
        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    return result


# ─── Episode runner ───────────────────────────────────────

async def run_episode(env, llm_client, tools, task_name: str, model_name: str) -> Dict[str, Any]:
    """Run a single incident triage episode with structured logging."""
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_name, model=model_name)

    try:
        reset_result = await env.reset(task_name=task_name)
        reset_obs = reset_result.observation if hasattr(reset_result, "observation") else reset_result
        metadata = getattr(reset_obs, "metadata", None) or {}
        briefing = metadata.get("briefing", f"An incident has been detected in task: {task_name}. Investigate immediately.")
        alerts = metadata.get("initial_alerts", "")

        # Bootstrap: if no alerts from reset, fetch them via tool call
        if not alerts:
            alert_result = await env.step(CallToolAction(tool_name="get_alerts", arguments={}))
            alert_obs = alert_result.observation
            alerts = str(alert_obs.result if hasattr(alert_obs, "result") else "")
            reward = alert_obs.reward or 0.0
            done = alert_obs.done or False
            rewards.append(reward)
            steps_taken = 1
            log_step(step=1, action="get_alerts()", reward=reward, done=done, error=None)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"INCIDENT BRIEFING:\n{briefing}\n\nINITIAL ALERTS:\n{alerts}"},
        ]

        obs = reset_obs
        step = steps_taken
        while not getattr(obs, "done", False) and step < MAX_STEPS:
            step += 1
            temp = 1.0 if "kimi" in model_name.lower() else 0.1

            # Retry on rate limits
            response = None
            for attempt in range(5):
                try:
                    response = llm_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        max_tokens=512,
                        temperature=temp,
                    )
                    break
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        wait = 2 ** attempt * 3
                        time.sleep(wait)
                    else:
                        raise

            if response is None:
                break

            msg = response.choices[0].message
            if not msg.tool_calls:
                break

            tc = msg.tool_calls[0]
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)
            tool_call_id = tc.id

            # Build assistant message
            assistant_msg = {
                "role": "assistant",
                "content": getattr(msg, "content", None),
                "tool_calls": [{"id": tool_call_id, "type": "function",
                               "function": {"name": tool_name, "arguments": tc.function.arguments}}],
            }
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            messages.append(assistant_msg)

            # Execute action
            action_str = f"{tool_name}({json.dumps(tool_args)})"
            action = CallToolAction(tool_name=tool_name, arguments=tool_args)
            step_result = await env.step(action)
            obs = step_result.observation

            reward = obs.reward or 0.0
            done = obs.done or False
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if not obs.done:
                result_text = str(obs.result if hasattr(obs, "result") else obs.metadata)
                messages.append({"role": "tool", "tool_call_id": tool_call_id,
                               "content": result_text[:2000]})

        # Compute final score
        final_reward = rewards[-1] if rewards else 0.0
        score = final_reward  # Terminal reward is the score
        success = score >= 0.5

        return {"task": task_name, "reward": final_reward, "steps": steps_taken, "metadata": getattr(obs, "metadata", {})}

    finally:
        log_end(success=success, steps=steps_taken, score=score if 'score' in dir() else 0.0,
                rewards=rewards)


async def main_async(base_url: str, preset: str = None, run_all: bool = False):
    # Single-model run (default or specific preset)
    api_base = API_BASE_URL
    api_key = API_KEY
    model = MODEL_NAME

    if not api_key:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable")
        sys.exit(1)

    llm_client = OpenAI(base_url=api_base, api_key=api_key)

    async with IncidentTriageEnv(base_url=base_url) as env:
        mcp_tools = await env.list_tools()
        tools = tools_to_openai_format(mcp_tools)

        results = []
        for task in TASKS:
            result = await run_episode(env, llm_client, tools, task, model)
            results.append(result)

        # Summary
        print(f"\n{'='*50}")
        print("BASELINE RESULTS")
        print(f"{'='*50}")
        for r in results:
            status = "PASS" if r["reward"] > 0.5 else "PARTIAL" if r["reward"] > 0 else "FAIL"
            print(f"  {r['task']}: {r['reward']:.4f} ({status}) [{r['steps']} steps]")
        avg = sum(r["reward"] for r in results) / len(results)
        print(f"  Average: {avg:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Incident Triage Environment — Baseline Inference")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Environment server URL")
    args = parser.parse_args()
    asyncio.run(main_async(args.base_url))


if __name__ == "__main__":
    main()

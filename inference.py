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
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openenv.core.env_server.mcp_types import CallToolAction

# Try local import first, fall back to installed package
try:
    from client import IncidentTriageEnv
except ImportError:
    try:
        from incident_triage_env.client import IncidentTriageEnv
    except ImportError:
        from openenv.core.mcp_client import MCPToolClient as IncidentTriageEnv

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
    safe_score = max(0.02, min(0.98, score))
    safe_rewards = [max(0.02, min(0.98, r)) for r in rewards] if rewards else [0.02]
    rewards_str = ",".join(f"{r:.2f}" for r in safe_rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={safe_score:.2f} rewards={rewards_str}", flush=True)


# ─── Tool conversion ─────────────────────────────────────

def tools_to_openai_format(mcp_tools) -> List[dict]:
    """Convert MCP tool definitions to OpenAI function-calling format."""
    result = []
    for tool in mcp_tools:
        props = {}
        required = []
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", {}) or {}
        if isinstance(schema, dict) and "properties" in schema:
            for name, field_schema in schema["properties"].items():
                props[name] = {
                    "type": field_schema.get("type", "string"),
                    "description": field_schema.get("description", ""),
                }
            required = schema.get("required", [])
        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
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
    score = 0.02

    log_start(task=task_name, model=model_name)

    try:
        # Reset the environment
        reset_result = await env.reset(task_name=task_name)

        # Extract observation — handle both StepResult wrapper and raw Observation
        if hasattr(reset_result, "observation"):
            reset_obs = reset_result.observation
        else:
            reset_obs = reset_result

        # Always fetch alerts as first step — most reliable way to get context
        # (WebSocket client path may lose custom observation fields from reset)
        alert_result = await env.step(CallToolAction(tool_name="get_alerts", arguments={}))
        alert_obs = alert_result.observation if hasattr(alert_result, "observation") else alert_result
        alerts_text = ""
        if hasattr(alert_obs, "result"):
            result = alert_obs.result
            if isinstance(result, str):
                alerts_text = result
            elif isinstance(result, dict):
                alerts_text = json.dumps(result, indent=2)
            elif hasattr(result, "data"):
                alerts_text = str(result.data)
            elif hasattr(result, "content"):
                # FastMCP wraps results in content list
                content = result.content if isinstance(result.content, list) else [result.content]
                texts = []
                for item in content:
                    if isinstance(item, dict):
                        texts.append(item.get("text", str(item)))
                    elif hasattr(item, "text"):
                        texts.append(item.text)
                    else:
                        texts.append(str(item))
                alerts_text = "\n".join(texts)
            else:
                alerts_text = str(result)

        reward = max(0.02, min(0.98, getattr(alert_obs, "reward", 0.02) or 0.02))
        done = getattr(alert_obs, "done", False) or False
        rewards.append(reward)
        steps_taken = 1
        log_step(step=1, action="get_alerts()", reward=reward, done=done, error=None)

        # Extract briefing from alert text or use default
        briefing = f"An incident has been detected in task: {task_name}. Investigate immediately."
        if alerts_text:
            try:
                alert_data = json.loads(alerts_text)
                if isinstance(alert_data, dict) and "briefing" in alert_data:
                    briefing = alert_data["briefing"]
                    alerts_text = json.dumps(alert_data.get("alerts", []), indent=2)
            except (json.JSONDecodeError, TypeError):
                pass

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"INCIDENT BRIEFING:\n{briefing}\n\nINITIAL ALERTS:\n{alerts_text}"},
        ]

        obs = alert_obs
        step = steps_taken
        while not getattr(obs, "done", False) and step < MAX_STEPS:
            step += 1
            temp = 0.1

            # LLM call with retry on rate limits
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
                        print(f"[DEBUG] Rate limited, waiting {wait}s", flush=True)
                        time.sleep(wait)
                    else:
                        print(f"[DEBUG] LLM error: {e}", flush=True)
                        raise

            if response is None:
                print("[DEBUG] LLM returned no response after retries", flush=True)
                break

            msg = response.choices[0].message
            if not msg.tool_calls:
                # No tool call — LLM may have finished or be confused
                print(f"[DEBUG] No tool calls in response, finishing", flush=True)
                break

            tc = msg.tool_calls[0]
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
            tool_call_id = tc.id

            # Build assistant message
            assistant_msg = {
                "role": "assistant",
                "content": getattr(msg, "content", None),
                "tool_calls": [{"id": tool_call_id, "type": "function",
                               "function": {"name": tool_name, "arguments": tc.function.arguments}}],
            }
            messages.append(assistant_msg)

            # Execute action
            action_str = f"{tool_name}({json.dumps(tool_args)})"
            try:
                action = CallToolAction(tool_name=tool_name, arguments=tool_args)
                step_result = await env.step(action)
                obs = step_result.observation if hasattr(step_result, "observation") else step_result
            except Exception as e:
                print(f"[DEBUG] Step error: {e}", flush=True)
                obs = type("FakeObs", (), {"done": False, "reward": 0.02, "result": str(e)})()

            reward = max(0.02, min(0.98, getattr(obs, "reward", 0.02) or 0.02))
            done = getattr(obs, "done", False) or False
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if not done:
                # Extract result text for LLM context
                result_text = ""
                if hasattr(obs, "result"):
                    result = obs.result
                    if isinstance(result, str):
                        result_text = result
                    elif isinstance(result, dict):
                        result_text = json.dumps(result, indent=2)
                    elif hasattr(result, "data"):
                        result_text = str(result.data)
                    elif hasattr(result, "content"):
                        content = result.content if isinstance(result.content, list) else [result.content]
                        texts = []
                        for item in content:
                            if isinstance(item, dict):
                                texts.append(item.get("text", str(item)))
                            elif hasattr(item, "text"):
                                texts.append(item.text)
                            else:
                                texts.append(str(item))
                        result_text = "\n".join(texts)
                    else:
                        result_text = str(result)
                elif hasattr(obs, "metadata") and obs.metadata:
                    result_text = json.dumps(obs.metadata, indent=2)
                else:
                    result_text = "Action completed."

                messages.append({"role": "tool", "tool_call_id": tool_call_id,
                               "content": result_text[:4000]})

        # Final score
        final_reward = rewards[-1] if rewards else 0.02
        score = max(0.02, min(0.98, final_reward))
        success = score >= 0.5

        return {"task": task_name, "reward": score, "steps": steps_taken, "metadata": {}}

    except Exception as e:
        print(f"[DEBUG] Episode error: {traceback.format_exc()}", flush=True)
        score = 0.02
        success = False
        return {"task": task_name, "reward": 0.02, "steps": steps_taken, "metadata": {}}

    finally:
        final_score = max(0.02, min(0.98, score))
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


async def main_async(base_url: str):
    api_base = API_BASE_URL
    api_key = API_KEY
    model = MODEL_NAME

    if not api_key:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable", flush=True)
        sys.exit(1)

    print(f"[DEBUG] Connecting to {base_url}", flush=True)
    print(f"[DEBUG] Using LLM: {model} at {api_base}", flush=True)

    llm_client = OpenAI(base_url=api_base, api_key=api_key)

    try:
        async with IncidentTriageEnv(base_url=base_url) as env:
            mcp_tools = await env.list_tools()
            tools = tools_to_openai_format(mcp_tools)
            print(f"[DEBUG] Discovered {len(tools)} tools", flush=True)

            results = []
            for task in TASKS:
                try:
                    result = await run_episode(env, llm_client, tools, task, model)
                except Exception as e:
                    print(f"[DEBUG] Task {task} failed: {traceback.format_exc()}", flush=True)
                    result = {"task": task, "reward": 0.02, "steps": 0, "metadata": {}}
                    log_start(task=task, model=model)
                    log_end(success=False, steps=0, score=0.02, rewards=[0.02])
                results.append(result)

            # Summary
            print(f"\n{'='*50}", flush=True)
            print("BASELINE RESULTS", flush=True)
            print(f"{'='*50}", flush=True)
            for r in results:
                status = "PASS" if r["reward"] > 0.5 else "PARTIAL" if r["reward"] > 0 else "FAIL"
                print(f"  {r['task']}: {r['reward']:.4f} ({status}) [{r['steps']} steps]", flush=True)
            avg = sum(r["reward"] for r in results) / len(results)
            print(f"  Average: {avg:.4f}", flush=True)
    except Exception as e:
        print(f"[DEBUG] Connection failed: {traceback.format_exc()}", flush=True)
        # Emit valid [END] markers for all tasks so evaluator can parse output
        for task in TASKS:
            log_start(task=task, model=model)
            log_end(success=False, steps=0, score=0.02, rewards=[0.02])
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Incident Triage Environment — Baseline Inference")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Environment server URL")
    args = parser.parse_args()
    asyncio.run(main_async(args.base_url))


if __name__ == "__main__":
    main()

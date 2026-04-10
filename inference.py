"""
Inference Script — Incident Response Triage Environment
========================================================
MANDATORY env vars:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=incident_triage_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Usage:
    python inference.py
    python inference.py --base-url http://localhost:8000
    python inference.py --base-url https://Farseen0-incident-triage-env.hf.space
"""
import argparse
import json
import os
import sys
import time
import traceback
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from openai import OpenAI

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


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    safe_score = max(0.02, min(0.98, score))
    safe_rewards = [max(0.02, min(0.98, r)) for r in rewards] if rewards else [0.02]
    rewards_str = ",".join(f"{r:.2f}" for r in safe_rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={safe_score:.2f} rewards={rewards_str}", flush=True)


# ─── HTTP helpers (no openenv dependency) ─────────────────

def http_post(url: str, data: dict) -> dict:
    """POST JSON to URL and return parsed response."""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get(url: str) -> dict:
    """GET URL and return parsed response."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def env_reset(base_url: str, task_name: str) -> dict:
    """Reset the environment via HTTP POST /reset."""
    return http_post(f"{base_url}/reset", {"task_name": task_name})


def env_step(base_url: str, tool_name: str, arguments: dict) -> dict:
    """Execute a tool call via HTTP POST /step."""
    return http_post(f"{base_url}/step", {
        "action": {
            "type": "call_tool",
            "tool_name": tool_name,
            "arguments": arguments,
        }
    })


def extract_result_text(obs: dict) -> str:
    """Extract readable text from a step observation."""
    result = obs.get("result")
    if result is None:
        return "No result."
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        # FastMCP wraps results in content list
        if "content" in result:
            content = result["content"]
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict):
                        texts.append(item.get("text", json.dumps(item)))
                    else:
                        texts.append(str(item))
                return "\n".join(texts)
            return str(content)
        if "data" in result:
            return str(result["data"])
        if "structured_content" in result:
            sc = result["structured_content"]
            return sc.get("result", json.dumps(sc))
        return json.dumps(result, indent=2)
    return str(result)


# ─── Tool schema discovery ────────────────────────────────

def discover_tools(base_url: str) -> List[dict]:
    """Discover available tools from the environment and convert to OpenAI format."""
    # Use HTTP /step with list_tools action
    try:
        resp = http_post(f"{base_url}/step", {
            "action": {"type": "list_tools"}
        })
        obs = resp.get("observation", {})
        tools_list = obs.get("tools", [])
    except Exception:
        # Fallback: hardcoded tool definitions (we know our own tools)
        tools_list = []

    if tools_list:
        result = []
        for tool in tools_list:
            schema = tool.get("input_schema", tool.get("inputSchema", {}))
            props = {}
            required = []
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
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": {"type": "object", "properties": props, "required": required},
                },
            })
        return result

    # Hardcoded fallback — our known tool definitions
    return [
        {"type": "function", "function": {"name": "get_alerts", "description": "Get all currently firing alerts for the incident, including the incident briefing.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "read_logs", "description": "Read recent log lines from a specific service.", "parameters": {"type": "object", "properties": {"service": {"type": "string", "description": "Service name"}, "lines": {"type": "integer", "description": "Number of log lines (default 50, max 200)"}}, "required": ["service"]}}},
        {"type": "function", "function": {"name": "check_metrics", "description": "Check performance metrics for a service.", "parameters": {"type": "object", "properties": {"service": {"type": "string", "description": "Service name"}, "metric": {"type": "string", "description": "Metric name or 'all'"}}, "required": ["service"]}}},
        {"type": "function", "function": {"name": "get_service_topology", "description": "Get the microservice dependency graph.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "set_severity", "description": "Classify the incident severity.", "parameters": {"type": "object", "properties": {"level": {"type": "string", "description": "P1, P2, P3, or P4"}}, "required": ["level"]}}},
        {"type": "function", "function": {"name": "diagnose", "description": "Submit root cause diagnosis.", "parameters": {"type": "object", "properties": {"root_cause_service": {"type": "string", "description": "Root cause service"}, "root_cause_category": {"type": "string", "description": "Root cause category"}}, "required": ["root_cause_service", "root_cause_category"]}}},
        {"type": "function", "function": {"name": "remediate", "description": "Take a remediation action.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "description": "Remediation action"}, "target_service": {"type": "string", "description": "Target service"}}, "required": ["action", "target_service"]}}},
        {"type": "function", "function": {"name": "submit_report", "description": "Finalize your incident report. Ends the episode.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    ]


# ─── Episode runner ───────────────────────────────────────

def run_episode(base_url: str, llm_client, tools: List[dict], task_name: str, model_name: str) -> Dict[str, Any]:
    """Run a single incident triage episode with structured logging."""
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.02

    log_start(task=task_name, model=model_name)

    try:
        # Reset the environment
        reset_data = env_reset(base_url, task_name)

        # Get alerts as first step (most reliable way to get context)
        alert_data = env_step(base_url, "get_alerts", {})
        alert_obs = alert_data.get("observation", {})
        alert_text = extract_result_text(alert_obs)
        reward = max(0.02, min(0.98, alert_data.get("reward") or 0.02))
        done = alert_data.get("done", False)
        rewards.append(reward)
        steps_taken = 1
        log_step(step=1, action="get_alerts()", reward=reward, done=done)

        # Extract briefing from alert response
        briefing = f"An incident has been detected in task: {task_name}. Investigate immediately."
        try:
            parsed = json.loads(alert_text) if isinstance(alert_text, str) else alert_text
            if isinstance(parsed, dict) and "briefing" in parsed:
                briefing = parsed["briefing"]
                alert_text = json.dumps(parsed.get("alerts", []), indent=2)
        except (json.JSONDecodeError, TypeError):
            pass

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"INCIDENT BRIEFING:\n{briefing}\n\nINITIAL ALERTS:\n{alert_text}"},
        ]

        step = steps_taken
        while not done and step < MAX_STEPS:
            step += 1

            # LLM call with retry
            response = None
            for attempt in range(5):
                try:
                    response = llm_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        max_tokens=512,
                        temperature=0.1,
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
                break

            msg = response.choices[0].message
            if not msg.tool_calls:
                break

            tc = msg.tool_calls[0]
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
            tool_call_id = tc.id

            # Add assistant message to conversation
            messages.append({
                "role": "assistant",
                "content": getattr(msg, "content", None),
                "tool_calls": [{"id": tool_call_id, "type": "function",
                               "function": {"name": tool_name, "arguments": tc.function.arguments}}],
            })

            # Execute action via HTTP
            action_str = f"{tool_name}({json.dumps(tool_args)})"
            try:
                step_data = env_step(base_url, tool_name, tool_args)
                obs = step_data.get("observation", {})
                reward = max(0.02, min(0.98, step_data.get("reward") or 0.02))
                done = step_data.get("done", False)
            except Exception as e:
                print(f"[DEBUG] Step error: {e}", flush=True)
                obs = {}
                reward = 0.02
                done = False

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done)

            if not done:
                result_text = extract_result_text(obs)
                messages.append({"role": "tool", "tool_call_id": tool_call_id,
                               "content": result_text[:4000]})

        # Final score
        final_reward = rewards[-1] if rewards else 0.02
        score = max(0.02, min(0.98, final_reward))
        success = score >= 0.5
        return {"task": task_name, "reward": score, "steps": steps_taken}

    except Exception as e:
        print(f"[DEBUG] Episode error: {traceback.format_exc()}", flush=True)
        score = 0.02
        success = False
        return {"task": task_name, "reward": 0.02, "steps": steps_taken}

    finally:
        final_score = max(0.02, min(0.98, score))
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


def main():
    parser = argparse.ArgumentParser(description="Incident Triage Environment — Baseline Inference")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Environment server URL")
    args = parser.parse_args()

    api_key = API_KEY
    model = MODEL_NAME

    if not api_key:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable", flush=True)
        sys.exit(1)

    base_url = args.base_url.rstrip("/")
    print(f"[DEBUG] Connecting to {base_url}", flush=True)
    print(f"[DEBUG] Using LLM: {model} at {API_BASE_URL}", flush=True)

    # Verify server is reachable
    try:
        health = http_get(f"{base_url}/health")
        print(f"[DEBUG] Server health: {health.get('status')}", flush=True)
    except Exception as e:
        print(f"[DEBUG] Server not reachable: {e}", flush=True)
        # Still emit valid output so evaluator can parse
        for task in TASKS:
            log_start(task=task, model=model)
            log_end(success=False, steps=0, score=0.02, rewards=[0.02])
        sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    # Discover tools
    tools = discover_tools(base_url)
    print(f"[DEBUG] Discovered {len(tools)} tools", flush=True)

    results = []
    for task in TASKS:
        try:
            result = run_episode(base_url, llm_client, tools, task, model)
        except Exception as e:
            print(f"[DEBUG] Task {task} failed: {traceback.format_exc()}", flush=True)
            result = {"task": task, "reward": 0.02, "steps": 0}
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


if __name__ == "__main__":
    main()

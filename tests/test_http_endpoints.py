"""HTTP endpoint tests — verifies the serialization path evaluators use.

These tests catch the critical bug where serialize_observation() strips
metadata from base Observation, returning empty {} to HTTP clients.

NOTE: HTTP mode in OpenEnv is stateless — each request creates a new env
instance. Multi-step episodes require WebSocket. These tests verify
individual endpoint responses are properly serialized.
"""
import pytest
from httpx import ASGITransport, AsyncClient

from incident_triage_env.server.app import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ── Health / Metadata / Schema ──


@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


@pytest.mark.anyio
async def test_metadata_populated(client):
    resp = await client.get("/metadata")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "IncidentTriageEnvironment"
    assert "incident" in data["description"].lower()
    assert data["author"] is not None
    assert data["version"] == "1.0.0"


@pytest.mark.anyio
async def test_schema_structure(client):
    resp = await client.get("/schema")
    assert resp.status_code == 200
    data = resp.json()
    assert "action" in data
    assert "observation" in data
    assert "state" in data


# ── Reset (THE critical test) ──


@pytest.mark.anyio
async def test_reset_observation_non_empty(client):
    """THE critical test: /reset must return non-empty observation."""
    resp = await client.post("/reset", json={})
    assert resp.status_code == 200
    data = resp.json()
    obs = data["observation"]
    # Must NOT be empty — this was the killer bug
    assert obs != {}, "Observation is empty — serialization bug!"
    assert "task_name" in obs
    assert obs["task_name"] != ""
    assert "briefing" in obs
    assert obs["briefing"] != ""
    assert "all_task_names" in obs
    assert len(obs["all_task_names"]) >= 3
    assert "available_tools" in obs
    assert "services" in obs
    assert "instructions" in obs


@pytest.mark.anyio
async def test_reset_reward_in_range(client):
    resp = await client.post("/reset", json={})
    data = resp.json()
    reward = data["reward"]
    assert 0 < reward < 1
    assert data["done"] is False


@pytest.mark.anyio
async def test_reset_has_all_fields(client):
    """Verify all expected fields are present in reset observation."""
    resp = await client.post("/reset", json={})
    obs = resp.json()["observation"]
    expected_fields = [
        "task_name", "all_task_names", "briefing", "initial_alerts",
        "available_tools", "severity_options", "root_cause_categories",
        "remediation_actions", "services", "instructions",
    ]
    for field in expected_fields:
        assert field in obs, f"Missing field: {field}"


# ── State ──


@pytest.mark.anyio
async def test_state_endpoint_responds(client):
    """State endpoint returns valid JSON with base fields.
    NOTE: HTTP mode creates a fresh env per request, so custom fields
    will have default values. WebSocket sessions have full state.
    """
    resp = await client.get("/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "step_count" in data
    assert isinstance(data["step_count"], int)


# ── Step ──


@pytest.mark.anyio
async def test_step_get_alerts(client):
    """Step with get_alerts returns non-empty observation.
    NOTE: HTTP mode is stateless — scenario may not be loaded.
    We verify the observation structure, not the content.
    """
    resp = await client.post("/step", json={
        "action": {
            "type": "call_tool",
            "tool_name": "get_alerts",
            "arguments": {}
        }
    })
    assert resp.status_code == 200
    data = resp.json()
    obs = data["observation"]
    assert obs != {}, "Step observation is empty"
    assert "tool_name" in obs
    assert obs["tool_name"] == "get_alerts"
    assert "result" in obs


@pytest.mark.anyio
async def test_step_json_string_arguments(client):
    """TriageCallToolAction should handle JSON string arguments."""
    resp = await client.post("/step", json={
        "action": {
            "type": "call_tool",
            "tool_name": "read_logs",
            "arguments": '{"service": "order-service", "lines": 10}'
        }
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["observation"] != {}


# ── Full episode via direct env (not HTTP) to verify terminal obs ──


def test_terminal_observation_type():
    """Verify terminal observation is IncidentTriageTerminalObservation
    with named fields (not base Observation with metadata)."""
    from incident_triage_env.server.incident_environment import IncidentTriageEnvironment
    from incident_triage_env.models import IncidentTriageTerminalObservation
    from openenv.core.env_server.mcp_types import CallToolAction

    env = IncidentTriageEnvironment()
    env.reset(task_name="easy_oom_crash")
    env.step(CallToolAction(tool_name="get_alerts", arguments={}))
    env.step(CallToolAction(tool_name="read_logs", arguments={"service": "order-service", "lines": 20}))
    env.step(CallToolAction(tool_name="check_metrics", arguments={"service": "order-service", "metric": "all"}))
    env.step(CallToolAction(tool_name="set_severity", arguments={"level": "P2"}))
    env.step(CallToolAction(tool_name="diagnose", arguments={"root_cause_service": "order-service", "root_cause_category": "memory_exhaustion"}))
    env.step(CallToolAction(tool_name="remediate", arguments={"action": "restart_service", "target_service": "order-service"}))
    obs = env.step(CallToolAction(tool_name="submit_report", arguments={}))

    assert obs.done is True
    assert isinstance(obs, IncidentTriageTerminalObservation)
    assert obs.severity_correct is True
    assert obs.service_correct is True
    assert obs.terminal_reward > 0
    assert "severity" in obs.grading_rubric


def test_reset_observation_type():
    """Verify reset observation is IncidentTriageResetObservation."""
    from incident_triage_env.server.incident_environment import IncidentTriageEnvironment
    from incident_triage_env.models import IncidentTriageResetObservation

    env = IncidentTriageEnvironment()
    obs = env.reset(task_name="easy_oom_crash")
    assert isinstance(obs, IncidentTriageResetObservation)
    assert obs.task_name == "easy_oom_crash"
    assert len(obs.all_task_names) == 5
    assert obs.briefing != ""
    assert len(obs.available_tools) == 8

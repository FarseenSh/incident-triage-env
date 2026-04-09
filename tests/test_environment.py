"""Tests for the Incident Triage Environment."""
import pytest
from incident_triage_env.server.incident_environment import IncidentTriageEnvironment
from incident_triage_env.models import TASK_NAMES, SERVICE_NAMES
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


@pytest.fixture
def env():
    return IncidentTriageEnvironment()


def do_step(env, tool_name, **kwargs):
    return env.step(CallToolAction(tool_name=tool_name, arguments=kwargs))


def assert_reward_in_range(reward):
    assert reward is not None, "Reward is None"
    assert 0 < reward < 1, f"Reward {reward} not in (0, 1)"


# ── Test 1: Reset returns valid observation ──

def test_reset_returns_valid_observation(env):
    obs = env.reset(task_name="easy_oom_crash")
    assert obs.done is False
    assert_reward_in_range(obs.reward)
    assert "briefing" in obs.metadata
    assert "task_name" in obs.metadata


# ── Test 2: All tasks reset successfully ──

def test_all_tasks_reset_successfully(env):
    for task in TASK_NAMES:
        obs = env.reset(task_name=task)
        assert obs.done is False
        assert obs.metadata["task_name"] == task


# ── Test 3: Correct answer scores high ──

def test_correct_answer_scores_high(env):
    env.reset(task_name="easy_oom_crash")
    do_step(env, "get_alerts")
    do_step(env, "read_logs", service="order-service", lines=20)
    do_step(env, "check_metrics", service="order-service", metric="all")
    do_step(env, "set_severity", level="P2")
    do_step(env, "diagnose", root_cause_service="order-service", root_cause_category="memory_exhaustion")
    do_step(env, "remediate", action="restart_service", target_service="order-service")
    obs = do_step(env, "submit_report")
    assert obs.done is True
    assert obs.reward >= 0.7
    assert obs.metadata["severity_correct"] is True
    assert obs.metadata["service_correct"] is True


# ── Test 4: Wrong answer scores low ──

def test_wrong_answer_scores_low(env):
    env.reset(task_name="easy_oom_crash")
    do_step(env, "set_severity", level="P4")
    do_step(env, "diagnose", root_cause_service="cache", root_cause_category="cpu_throttling")
    do_step(env, "remediate", action="clear_cache", target_service="cache")
    obs = do_step(env, "submit_report")
    assert obs.done is True
    assert obs.reward < 0.4


# ── Test 5: All rewards strictly in (0, 1) ──

def test_rewards_strictly_in_range(env):
    for task in TASK_NAMES:
        env.reset(task_name=task)
        # Take a few steps
        obs = do_step(env, "get_alerts")
        assert_reward_in_range(obs.reward)
        obs = do_step(env, "read_logs", service="order-service", lines=5)
        assert_reward_in_range(obs.reward)
        # Submit
        do_step(env, "set_severity", level="P2")
        do_step(env, "diagnose", root_cause_service="order-service", root_cause_category="memory_exhaustion")
        do_step(env, "remediate", action="restart_service", target_service="order-service")
        obs = do_step(env, "submit_report")
        assert_reward_in_range(obs.reward)


# ── Test 6: Submit without diagnosis ──

def test_submit_without_diagnosis(env):
    env.reset(task_name="easy_oom_crash")
    obs = do_step(env, "submit_report")
    assert obs.done is True
    assert_reward_in_range(obs.reward)
    assert obs.reward < 0.3


# ── Test 7: Invalid tool arguments handled gracefully ──

def test_invalid_tool_graceful(env):
    env.reset(task_name="easy_oom_crash")
    obs = do_step(env, "read_logs", service="nonexistent-service", lines=10)
    assert obs.done is False
    assert_reward_in_range(obs.reward)


# ── Test 8: Max steps terminates episode ──

def test_max_steps_terminates(env):
    env.reset(task_name="easy_oom_crash")
    for i in range(21):
        obs = do_step(env, "get_alerts")
        if obs.done:
            break
    assert obs.done is True
    assert_reward_in_range(obs.reward)


# ── Test 9: State is sanitized (no ground truth leaked) ──

def test_state_sanitized(env):
    env.reset(task_name="easy_oom_crash")
    state = env.state
    assert state.ground_truth_severity == ""
    assert state.ground_truth_root_cause_service == ""
    assert state.ground_truth_root_cause_category == ""
    assert state.ground_truth_remediation_action == ""
    assert state.ground_truth_remediation_target == ""
    assert state.ground_truth_affected_services == []


# ── Test 10: Repeat action penalty ──

def test_repeat_penalty(env):
    env.reset(task_name="easy_oom_crash")
    do_step(env, "read_logs", service="order-service", lines=50)
    do_step(env, "read_logs", service="order-service", lines=50)  # duplicate
    assert env._state.repeated_actions == 1
    assert env._state.penalty <= -0.02


# ── Test 11: Workflow bonus ──

def test_workflow_bonus(env):
    # Correct workflow: investigate first, then diagnose
    env.reset(task_name="easy_oom_crash")
    do_step(env, "read_logs", service="order-service", lines=20)
    do_step(env, "check_metrics", service="order-service", metric="all")
    do_step(env, "set_severity", level="P2")  # triggers workflow bonus
    assert env._state.workflow_bonus == 0.03
    assert env._state.workflow_violations == 0


def test_workflow_violation(env):
    # Wrong workflow: diagnose without investigating
    env.reset(task_name="easy_oom_crash")
    do_step(env, "diagnose", root_cause_service="order-service", root_cause_category="memory_exhaustion")
    assert env._state.workflow_violations >= 1
    assert env._state.penalty <= -0.02

import json
import logging
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import Action, Observation

from ..models import (
    AVAILABLE_TOOLS, SEVERITY_LEVELS, ROOT_CAUSE_CATEGORIES,
    REMEDIATION_ACTIONS, SERVICE_NAMES, TASK_NAMES, IncidentTriageState,
)
from ..scenarios import ScenarioRegistry
from ..simulation.service_graph import ServiceGraph

logger = logging.getLogger(__name__)

MAX_STEPS = 20  # Environment max — inference.py should match or be lower


class IncidentTriageEnvironment(MCPEnvironment):
    """SRE incident response triage environment using MCP tools."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("incident_triage_env")

        # ── Register 8 MCP tools with input validation ──────────────

        @mcp.tool
        def get_alerts() -> str:
            """Get all currently firing alerts for the incident, including the incident briefing."""
            if self._current_scenario is None:
                return json.dumps({"error": "No active incident. Call reset() first."})
            briefing = self._current_scenario.get_initial_observation_text()
            alerts = self._current_scenario.get_alerts()
            return json.dumps({"briefing": briefing, "alerts": alerts}, indent=2)

        @mcp.tool
        def read_logs(service: str, lines: int = 50) -> str:
            """Read recent log lines from a specific service.
            Args:
                service: Service name (e.g., 'order-service', 'database')
                lines: Number of log lines to return (default 50, max 200)
            """
            if service not in SERVICE_NAMES:
                return f"Error: Unknown service '{service}'. Valid services: {SERVICE_NAMES}"
            lines = min(max(1, lines), 200)
            if service not in self._state.services_investigated:
                self._state.services_investigated.append(service)
            logs = self._current_scenario.get_logs(service, lines)
            return "\n".join(logs)

        @mcp.tool
        def check_metrics(service: str, metric: str = "all") -> str:
            """Check performance metrics for a service.
            Args:
                service: Service name
                metric: Specific metric (cpu, memory, latency, error_rate, connections) or 'all'
            """
            if service not in SERVICE_NAMES:
                return f"Error: Unknown service '{service}'. Valid services: {SERVICE_NAMES}"
            valid_metrics = ["cpu", "memory", "latency", "error_rate", "connections", "all"]
            if metric not in valid_metrics:
                return f"Error: Unknown metric '{metric}'. Valid: {valid_metrics}"
            if service not in self._state.services_investigated:
                self._state.services_investigated.append(service)
            metrics = self._current_scenario.get_metrics(service, metric)
            return json.dumps(metrics, indent=2)

        @mcp.tool
        def get_service_topology() -> str:
            """Get the microservice dependency graph showing which services depend on which."""
            return self._service_graph.get_topology_description()

        @mcp.tool
        def set_severity(level: str) -> str:
            """Classify the incident severity.
            Args:
                level: Severity level — must be one of: P1, P2, P3, P4
            """
            level = level.upper().strip()
            if level not in SEVERITY_LEVELS:
                return f"Error: Invalid severity '{level}'. Must be one of: {SEVERITY_LEVELS}"
            self._state.agent_severity = level
            self._state.severity_set = level
            return f"Severity set to {level}"

        @mcp.tool
        def diagnose(root_cause_service: str, root_cause_category: str) -> str:
            """Submit your diagnosis of the root cause.
            Args:
                root_cause_service: The service where the root cause originates
                root_cause_category: Category — must be one of: memory_exhaustion, connection_pool_exhaustion, clock_skew, disk_full, cpu_throttling, network_partition, config_error, dependency_failure
            """
            if root_cause_service not in SERVICE_NAMES:
                return f"Error: Unknown service '{root_cause_service}'. Valid: {SERVICE_NAMES}"
            if root_cause_category not in ROOT_CAUSE_CATEGORIES:
                return f"Error: Unknown category '{root_cause_category}'. Valid: {ROOT_CAUSE_CATEGORIES}"
            self._state.agent_root_cause_service = root_cause_service
            self._state.agent_root_cause_category = root_cause_category
            self._state.diagnosis_submitted = True
            return f"Diagnosis recorded: {root_cause_category} on {root_cause_service}"

        @mcp.tool
        def remediate(action: str, target_service: str) -> str:
            """Take a remediation action on a service.
            Args:
                action: Action — must be one of: restart_service, scale_up, rollback_deploy, config_change, failover, clear_cache, increase_pool_size
                target_service: The service to apply the action to
            """
            if action not in REMEDIATION_ACTIONS:
                return f"Error: Unknown action '{action}'. Valid: {REMEDIATION_ACTIONS}"
            if target_service not in SERVICE_NAMES:
                return f"Error: Unknown service '{target_service}'. Valid: {SERVICE_NAMES}"
            self._state.agent_remediation_action = action
            self._state.agent_remediation_target = target_service
            self._state.remediation_submitted = True

            # Penalty for destructive action on healthy service
            gt = self._current_scenario.get_ground_truth()
            if target_service not in gt.affected_services and action == "restart_service":
                self._state.penalty -= 0.03
                return f"WARNING: Restarted healthy service {target_service}. Penalty applied."
            return f"Remediation applied: {action} on {target_service}"

        @mcp.tool
        def submit_report() -> str:
            """Finalize your incident report. This ends the episode and triggers grading.
            Make sure you have: set severity, submitted diagnosis, and applied remediation before calling this.
            """
            missing = []
            if not self._state.severity_set:
                missing.append("severity (call set_severity)")
            if not self._state.diagnosis_submitted:
                missing.append("diagnosis (call diagnose)")
            if not self._state.remediation_submitted:
                missing.append("remediation (call remediate)")
            if missing:
                return f"WARNING: Submitting incomplete report. Missing: {', '.join(missing)}. Report submitted anyway."
            self._state.report_submitted = True
            return "Report submitted. Episode complete."

        # ── Initialize base class ──────────────────────────────────
        super().__init__(mcp)
        self._service_graph = ServiceGraph()
        self._current_scenario = None
        self._state = IncidentTriageState()

    # ── reset() ────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        task_name = kwargs.get("task_name", "easy_oom_crash")
        if task_name not in TASK_NAMES:
            task_name = "easy_oom_crash"

        self._current_scenario = ScenarioRegistry.get(task_name, seed=seed)
        gt = self._current_scenario.get_ground_truth()

        self._state = IncidentTriageState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=task_name,
            ground_truth_severity=gt.severity,
            ground_truth_root_cause_service=gt.root_cause_service,
            ground_truth_root_cause_category=gt.root_cause_category,
            ground_truth_remediation_action=gt.remediation_action,
            ground_truth_remediation_target=gt.remediation_target,
            ground_truth_affected_services=gt.affected_services,
        )

        initial_text = self._current_scenario.get_initial_observation_text()
        alert_summary = json.dumps(self._current_scenario.get_alerts()[:3], indent=2)

        logger.info(f"Reset episode {self._state.episode_id} with task: {task_name}")

        return Observation(
            done=False,
            reward=0.01,
            metadata={
                "task_name": task_name,
                "all_task_names": TASK_NAMES,  # Validators enumerate tasks from this
                "briefing": initial_text,
                "initial_alerts": alert_summary,
                "available_tools": AVAILABLE_TOOLS,
                "severity_options": SEVERITY_LEVELS,
                "root_cause_categories": ROOT_CAUSE_CATEGORIES,
                "remediation_actions": REMEDIATION_ACTIONS,
                "services": SERVICE_NAMES,
                "instructions": (
                    "You are an SRE on-call. An incident has been detected. "
                    "Use the available tools to investigate, diagnose the root cause, "
                    "set the severity, apply remediation, and submit your report. "
                    "Available tools: get_alerts, read_logs, check_metrics, "
                    "get_service_topology, set_severity, diagnose, remediate, submit_report."
                ),
            },
        )

    # ── _step_impl() — REQUIRED by MCPEnvironment (abstract) ──────
    # This handles non-MCP actions. Our env is MCP-only, so return error.
    # WITHOUT THIS METHOD, the class CANNOT be instantiated (TypeError).

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions. Returns error since this env is MCP-only."""
        return Observation(
            done=False,
            reward=0.01,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    # ── Core step logic (shared by step and step_async) ───────────

    def _process_step_result(
        self, action: Action, obs: Observation
    ) -> Observation:
        """Shared logic for processing a step result. Called by both step() and step_async()."""
        gt = self._current_scenario.get_ground_truth()
        step_reward = 0.0

        if isinstance(action, CallToolAction):
            # Per-step investigation reward
            if action.tool_name in ("read_logs", "check_metrics"):
                service = action.arguments.get("service", "")
                if service in gt.affected_services or service in gt.causal_chain:
                    step_reward = 0.02
            elif action.tool_name == "get_service_topology" and "topology" not in self._state.actions_taken:
                self._state.actions_taken.append("topology")
                step_reward = 0.01
            elif action.tool_name == "get_alerts" and "alerts" not in self._state.actions_taken:
                self._state.actions_taken.append("alerts")
                step_reward = 0.01

            # Terminal: submit_report
            if action.tool_name == "submit_report":
                terminal_reward = self._compute_terminal_reward()
                total_reward = max(0.01, min(0.99,
                    terminal_reward
                    + self._state.investigation_reward
                    + self._state.penalty
                ))
                logger.info(
                    f"Episode {self._state.episode_id} ended: "
                    f"terminal={terminal_reward:.2f}, investigation={self._state.investigation_reward:.2f}, "
                    f"penalty={self._state.penalty:.2f}, total={total_reward:.4f}"
                )
                return Observation(
                    done=True,
                    reward=round(total_reward, 4),
                    metadata={
                        "terminal_reward": round(terminal_reward, 4),
                        "investigation_reward": round(self._state.investigation_reward, 4),
                        "penalty": round(self._state.penalty, 4),
                        "severity_correct": self._state.agent_severity == gt.severity,
                        "service_correct": self._state.agent_root_cause_service == gt.root_cause_service,
                        "category_correct": self._state.agent_root_cause_category == gt.root_cause_category,
                        "remediation_correct": (
                            self._state.agent_remediation_action == gt.remediation_action
                            and self._state.agent_remediation_target == gt.remediation_target
                        ),
                    },
                )

        self._state.investigation_reward += step_reward

        # Max steps check
        if self._state.step_count >= MAX_STEPS:
            terminal_reward = self._compute_terminal_reward()
            total_reward = max(0.01, min(0.99,
                terminal_reward + self._state.investigation_reward + self._state.penalty - 0.1
            ))
            logger.info(f"Episode {self._state.episode_id} terminated: max steps reached")
            return Observation(
                done=True,
                reward=round(total_reward, 4),
                metadata={
                    "error": f"Max steps ({MAX_STEPS}) reached.",
                    "terminal_reward": round(terminal_reward, 4),
                },
            )

        # Non-terminal: preserve tool result but ensure reward is in (0, 1)
        if hasattr(obs, 'reward'):
            obs.reward = 0.01
        return obs

    # ── step() — sync path ────────────────────────────────────────

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a step. Delegates MCP routing to base class, then applies custom logic."""
        # Only count CallToolAction as real steps — ListToolsAction is free
        if isinstance(action, CallToolAction):
            self._state.step_count += 1
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        # ListToolsAction: pass through but ensure reward is in (0, 1)
        if not isinstance(action, CallToolAction):
            if hasattr(obs, 'reward'):
                obs.reward = 0.01
            return obs
        return self._process_step_result(action, obs)

    # ── step_async() — WebSocket path (MUST be fully implemented) ─
    # The MCPToolClient connects via WebSocket. The WS handler calls
    # step_async() directly. If this is missing or incomplete, the
    # client path receives no rewards and episodes never terminate.

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async step for WebSocket handler. Mirrors step() exactly."""
        if isinstance(action, CallToolAction):
            self._state.step_count += 1
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        if not isinstance(action, CallToolAction):
            if hasattr(obs, 'reward'):
                obs.reward = 0.01
            return obs
        return self._process_step_result(action, obs)

    # ── Terminal reward computation ───────────────────────────────

    def _compute_terminal_reward(self) -> float:
        """Compute decomposed terminal reward (4 components, max 1.0)."""
        gt = self._current_scenario.get_ground_truth()
        score = 0.0

        # Severity: 0.2 max
        if self._state.agent_severity == gt.severity:
            score += 0.2
        elif self._state.agent_severity and self._state.agent_severity in SEVERITY_LEVELS:
            if abs(SEVERITY_LEVELS.index(self._state.agent_severity)
                   - SEVERITY_LEVELS.index(gt.severity)) == 1:
                score += 0.1  # Partial credit for off-by-one

        # Service identification: 0.2 max
        if self._state.agent_root_cause_service == gt.root_cause_service:
            score += 0.2
        elif self._state.agent_root_cause_service in gt.affected_services:
            score += 0.1  # Partial: identified affected but not root

        # Root cause category: 0.3 max
        if self._state.agent_root_cause_category == gt.root_cause_category:
            score += 0.3

        # Remediation: 0.3 max
        if (self._state.agent_remediation_action == gt.remediation_action
                and self._state.agent_remediation_target == gt.remediation_target):
            score += 0.3
        elif self._state.agent_remediation_action == gt.remediation_action:
            score += 0.1  # Partial: right action, wrong target

        return score

    @property
    def state(self) -> IncidentTriageState:
        """Get the current environment state.
        IMPORTANT: Returns a sanitized copy that EXCLUDES ground truth fields.
        This prevents reward hacking — agents cannot call state() to read answers.
        The full state (with ground truth) is only used internally for grading.
        """
        # Return a copy with ground truth fields blanked out
        sanitized = self._state.model_copy()
        sanitized.ground_truth_severity = ""
        sanitized.ground_truth_root_cause_service = ""
        sanitized.ground_truth_root_cause_category = ""
        sanitized.ground_truth_remediation_action = ""
        sanitized.ground_truth_remediation_target = ""
        sanitized.ground_truth_affected_services = []
        return sanitized

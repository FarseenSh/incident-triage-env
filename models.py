"""
State and observation types for the Incident Response Triage environment.

This environment uses the MCP protocol for tool interactions.
Use CallToolAction and ListToolsAction from openenv.core.env_server.mcp_types.
"""
from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Observation, State


AVAILABLE_TOOLS = [
    "get_alerts",
    "read_logs",
    "check_metrics",
    "get_service_topology",
    "set_severity",
    "diagnose",
    "remediate",
    "submit_report",
]

# Enums for deterministic grading (agent must select from these)
SEVERITY_LEVELS = ["P1", "P2", "P3", "P4"]
ROOT_CAUSE_CATEGORIES = [
    "memory_exhaustion",
    "connection_pool_exhaustion",
    "clock_skew",
    "disk_full",
    "cpu_throttling",
    "network_partition",
    "config_error",
    "dependency_failure",
]
REMEDIATION_ACTIONS = [
    "restart_service",
    "scale_up",
    "rollback_deploy",
    "config_change",
    "failover",
    "clear_cache",
    "increase_pool_size",
]
SERVICE_NAMES = [
    "api-gateway",
    "auth-service",
    "order-service",
    "inventory-service",
    "payment-service",
    "database",
    "cache",
    "message-queue",
]
TASK_NAMES = [
    "easy_oom_crash",
    "medium_cascade",
    "medium_disk_full",
    "hard_intermittent",
    "hard_network_partition",
]


class IncidentTriageState(State):
    """Internal environment state — tracks episode progress and hidden ground truth."""
    # Visible to state() endpoint
    task_name: str = ""
    services_investigated: List[str] = []
    actions_taken: List[str] = []
    severity_set: Optional[str] = None
    diagnosis_submitted: bool = False
    remediation_submitted: bool = False
    report_submitted: bool = False

    # Hidden ground truth — NOT exposed in observations
    ground_truth_severity: str = ""
    ground_truth_root_cause_service: str = ""
    ground_truth_root_cause_category: str = ""
    ground_truth_remediation_action: str = ""
    ground_truth_remediation_target: str = ""
    ground_truth_affected_services: List[str] = []

    # Agent's submitted answers (for grading)
    agent_severity: Optional[str] = None
    agent_root_cause_service: Optional[str] = None
    agent_root_cause_category: Optional[str] = None
    agent_remediation_action: Optional[str] = None
    agent_remediation_target: Optional[str] = None

    # Reward tracking
    investigation_reward: float = 0.0
    penalty: float = 0.0

    # Workflow phase tracking
    workflow_phase: str = "INVESTIGATE"
    workflow_bonus: float = 0.0
    workflow_violations: int = 0

    # Repeat action tracking
    repeated_actions: int = 0

    # Efficiency tracking
    efficiency_bonus: float = 0.0


# ── Custom observation types ──────────────────────────────────────
# openenv-core's serialize_observation() excludes done, reward, AND metadata
# from the HTTP response dict. Base Observation only has those 3 fields,
# so the HTTP response is {"observation": {}, ...} — empty.
# Custom subclass fields SURVIVE exclusion (like CallToolObservation's
# tool_name/result/error do for the step path).


class IncidentTriageResetObservation(Observation):
    """Observation returned by reset(). Named fields survive HTTP serialization."""
    task_name: str = Field(default="", description="Current task/scenario name")
    all_task_names: List[str] = Field(default_factory=list, description="All available task names")
    briefing: str = Field(default="", description="Incident briefing text")
    initial_alerts: str = Field(default="", description="Initial alert summary JSON")
    available_tools: List[str] = Field(default_factory=list, description="Available MCP tool names")
    severity_options: List[str] = Field(default_factory=list, description="Valid severity levels")
    root_cause_categories: List[str] = Field(default_factory=list, description="Valid root cause categories")
    remediation_actions: List[str] = Field(default_factory=list, description="Valid remediation actions")
    services: List[str] = Field(default_factory=list, description="Valid service names")
    instructions: str = Field(default="", description="Agent instructions")


class IncidentTriageTerminalObservation(Observation):
    """Observation returned on episode termination (submit_report / max steps)."""
    terminal_reward: float = Field(default=0.0, description="Terminal component of reward")
    investigation_reward: float = Field(default=0.0, description="Investigation component")
    penalty: float = Field(default=0.0, description="Penalty component")
    severity_correct: bool = Field(default=False, description="Whether severity was correct")
    service_correct: bool = Field(default=False, description="Whether root cause service was correct")
    category_correct: bool = Field(default=False, description="Whether root cause category was correct")
    remediation_correct: bool = Field(default=False, description="Whether remediation was correct")
    grading_rubric: Dict[str, Any] = Field(default_factory=dict, description="Grading rubric breakdown")
    investigation_summary: Dict[str, Any] = Field(default_factory=dict, description="Investigation details")
    workflow_score: Dict[str, Any] = Field(default_factory=dict, description="Workflow scoring details")
    efficiency_score: float = Field(default=0.0, description="Efficiency bonus")
    error: Optional[str] = Field(default=None, description="Error message if applicable")

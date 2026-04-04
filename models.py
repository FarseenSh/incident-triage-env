"""
State types for the Incident Response Triage environment.

This environment uses the MCP protocol for tool interactions.
Use CallToolAction and ListToolsAction from openenv.core.env_server.mcp_types.
"""
from typing import Dict, List, Optional
from openenv.core.env_server.types import State


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

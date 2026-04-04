from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class GroundTruth:
    severity: str                    # P1/P2/P3/P4
    root_cause_service: str          # e.g., "order-service"
    root_cause_category: str         # from ROOT_CAUSE_CATEGORIES enum
    remediation_action: str          # from REMEDIATION_ACTIONS enum
    remediation_target: str          # service name to apply action to
    affected_services: List[str]     # all services in the causal chain
    causal_chain: List[str]          # ordered: root → ... → symptom


class ScenarioBase(ABC):
    """Base class for incident scenarios."""

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed

    @abstractmethod
    def get_ground_truth(self) -> GroundTruth:
        """Return the hidden ground truth for grading."""

    @abstractmethod
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Return alerts that fire for this scenario."""

    @abstractmethod
    def get_logs(self, service: str, lines: int) -> List[str]:
        """Return log lines for a specific service."""

    @abstractmethod
    def get_metrics(self, service: str, metric: str) -> Dict[str, Any]:
        """Return metrics for a specific service."""

    @abstractmethod
    def get_initial_observation_text(self) -> str:
        """Return the initial briefing the agent sees after reset."""


class ScenarioRegistry:
    """Registry mapping task names to scenario classes.

    Usage:
        ScenarioRegistry.register("easy_oom_crash", EasyOOMCrash)
        scenario = ScenarioRegistry.get("easy_oom_crash", seed=42)
    """
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, task_name: str, scenario_class: type) -> None:
        """Register a scenario class under a task name."""
        cls._registry[task_name] = scenario_class

    @classmethod
    def get(cls, task_name: str, seed: Optional[int] = None) -> ScenarioBase:
        """Create and return a scenario instance by task name."""
        if task_name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown task: '{task_name}'. Available: {available}")
        return cls._registry[task_name](seed=seed)

    @classmethod
    def list_tasks(cls) -> List[str]:
        """Return all registered task names."""
        return list(cls._registry.keys())

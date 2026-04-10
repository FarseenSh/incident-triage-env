"""Task 1: Single Service OOM Kill — order-service runs out of memory."""
from typing import Any, Dict, List, Optional

from .base import ScenarioBase, GroundTruth
from ..simulation.log_generator import LogGenerator
from ..simulation.metrics_store import MetricsStore
from ..simulation.alert_engine import AlertEngine


# Error patterns — clear OOM signals but agent must infer remediation
_ORDER_SERVICE_ERRORS = [
    {"level": "WARN", "message": "Memory usage at 85% — approaching heap limit"},
    {"level": "WARN", "message": "GC overhead limit exceeded — long pause (4200ms)"},
    {"level": "ERROR", "message": "Failed to allocate 256MB — heap space exhausted"},
    {"level": "FATAL", "message": "java.lang.OutOfMemoryError: Java heap space"},
    {"level": "FATAL", "message": "Process killed by OOM killer (exit code 137)"},
    {"level": "ERROR", "message": "Service order-service terminated unexpectedly (signal 9)"},
    {"level": "ERROR", "message": "All in-flight requests dropped — connection pool drained"},
]

_GATEWAY_ERRORS = [
    {"level": "ERROR", "message": "upstream connection refused: order-service:8080"},
    {"level": "WARN", "message": "Circuit breaker OPEN for order-service"},
]


class EasyOOMCrash(ScenarioBase):
    """Single service OOM crash on order-service.

    Designed to be very easy — alerts, logs, and metrics all point clearly
    to order-service memory exhaustion. Severity P2 is explicit in alerts.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)
        self._log_gen = LogGenerator(seed=seed)
        self._metrics = MetricsStore(seed=seed)
        self._alerts = AlertEngine()

        # Set anomalous metrics for order-service
        self._metrics.set_metrics("order-service", {
            "memory": 98.0,
            "cpu": 85.0,
            "error_rate": 100.0,
            "latency": 99999.0,
        })

        # Alerts — clear signals but agent must reason about cause + fix
        self._alerts.create_alert(
            "CRITICAL", "order-service",
            "Process terminated by OOM killer (exit code 137)",
            "oom_killer",
        )
        self._alerts.create_alert(
            "HIGH", "order-service",
            "health check failed — service unresponsive after repeated retries",
            "health_check_failure",
        )

    def get_ground_truth(self) -> GroundTruth:
        return GroundTruth(
            severity="P2",
            root_cause_service="order-service",
            root_cause_category="memory_exhaustion",
            remediation_action="restart_service",
            remediation_target="order-service",
            affected_services=["order-service", "api-gateway"],
            causal_chain=["order-service", "api-gateway"],
        )

    def get_alerts(self) -> List[Dict[str, Any]]:
        return self._alerts.get_alerts()

    def get_logs(self, service: str, lines: int) -> List[str]:
        if service == "order-service":
            # High error ratio — OOM errors dominate the logs
            return self._log_gen.generate_logs(service, lines, _ORDER_SERVICE_ERRORS, error_ratio=0.5)
        if service == "api-gateway":
            return self._log_gen.generate_logs(service, lines, _GATEWAY_ERRORS, error_ratio=0.25)
        return self._log_gen.generate_normal_logs(service, lines)

    def get_metrics(self, service: str, metric: str) -> Dict[str, Any]:
        return self._metrics.get_metrics(service, metric)

    def get_initial_observation_text(self) -> str:
        return (
            "INCIDENT ALERT: order-service is DOWN. OOM killer triggered (exit code 137). "
            "API gateway reporting upstream connection failures. Customer-facing order "
            "endpoints returning 503 errors. Multiple services affected."
        )

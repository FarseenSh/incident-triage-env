"""Task 4: Disk Full on Database — writes fail, reads still work (partially)."""
from typing import Any, Dict, List, Optional

from .base import ScenarioBase, GroundTruth
from ..simulation.log_generator import LogGenerator
from ..simulation.metrics_store import MetricsStore
from ..simulation.alert_engine import AlertEngine


_DATABASE_ERRORS = [
    {"level": "ERROR", "message": "FATAL: could not write to file 'pg_wal/000000010000000000000042': No space left on device"},
    {"level": "ERROR", "message": "LOG: checkpoint failed: write error on WAL segment"},
    {"level": "WARN", "message": "disk usage: /var/lib/postgresql/data at 99.2% (47.6GB / 48GB)"},
    {"level": "ERROR", "message": "INSERT failed: could not extend relation — no space left on device"},
]

_ORDER_SERVICE_ERRORS = [
    {"level": "ERROR", "message": "database write failed: internal error"},
    {"level": "ERROR", "message": "order creation failed: could not persist to database"},
    {"level": "WARN", "message": "read-only mode activated — writes failing"},
    {"level": "INFO", "message": "order reads still functioning (cache + read replica)"},
]

_GATEWAY_ERRORS = [
    {"level": "ERROR", "message": "POST /api/v1/orders returning 500"},
    {"level": "WARN", "message": "GET requests succeeding, POST/PUT failing"},
]

# Red herring: payment-service shows errors too (downstream of order-service)
_PAYMENT_ERRORS = [
    {"level": "ERROR", "message": "payment record persistence failed"},
    {"level": "WARN", "message": "transaction rollback — upstream write failure"},
]


class MediumDiskFull(ScenarioBase):
    """Database disk full — writes fail but reads work, creating a confusing partial outage.

    The trap: order-service errors look like the app is broken. payment-service
    also fails. But GET requests work fine — only writes fail. The agent must
    notice the asymmetric failure pattern and trace it to database disk.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)
        self._log_gen = LogGenerator(seed=seed)
        self._metrics = MetricsStore(seed=seed)
        self._alerts = AlertEngine()

        self._metrics.set_metrics("database", {
            "cpu": 40.0,
            "memory": 72.0,
            "error_rate": 35.0,
            "latency": 500.0,
            "connections": 55,
        })
        self._metrics.set_metrics("order-service", {
            "error_rate": 45.0,
            "latency": 1200.0,
        })
        self._metrics.set_metrics("payment-service", {
            "error_rate": 30.0,
        })

        self._alerts.create_alert(
            "CRITICAL", "order-service",
            "order creation endpoint returning 500s", "order_write_failure",
        )
        self._alerts.create_alert(
            "HIGH", "payment-service",
            "payment persistence failures", "payment_write_failure",
        )
        self._alerts.create_alert(
            "HIGH", "database",
            "write operations failing — possible storage issue", "db_write_error",
        )
        self._alerts.create_alert(
            "WARN", "api-gateway",
            "POST/PUT requests failing, GET requests OK", "asymmetric_errors",
        )

    def get_ground_truth(self) -> GroundTruth:
        return GroundTruth(
            severity="P1",
            root_cause_service="database",
            root_cause_category="disk_full",
            remediation_action="scale_up",
            remediation_target="database",
            affected_services=["database", "order-service", "payment-service"],
            causal_chain=["database", "order-service", "api-gateway"],
        )

    def get_alerts(self) -> List[Dict[str, Any]]:
        return self._alerts.get_alerts()

    def get_logs(self, service: str, lines: int) -> List[str]:
        if service == "database":
            return self._log_gen.generate_logs(service, lines, _DATABASE_ERRORS, error_ratio=0.3)
        if service == "order-service":
            return self._log_gen.generate_logs(service, lines, _ORDER_SERVICE_ERRORS, error_ratio=0.3)
        if service == "api-gateway":
            return self._log_gen.generate_logs(service, lines, _GATEWAY_ERRORS, error_ratio=0.2)
        if service == "payment-service":
            return self._log_gen.generate_logs(service, lines, _PAYMENT_ERRORS, error_ratio=0.25)
        return self._log_gen.generate_normal_logs(service, lines)

    def get_metrics(self, service: str, metric: str) -> Dict[str, Any]:
        return self._metrics.get_metrics(service, metric)

    def get_initial_observation_text(self) -> str:
        return (
            "INCIDENT ALERT: Partial service outage. Order creation and payment "
            "processing are failing, but read operations still work. "
            "Asymmetric failure pattern detected — investigate storage and "
            "write paths."
        )

"""Task 2: Cascading Dependency Failure — database connection pool exhaustion."""
from typing import Any, Dict, List, Optional

from .base import ScenarioBase, GroundTruth
from ..simulation.log_generator import LogGenerator
from ..simulation.metrics_store import MetricsStore
from ..simulation.alert_engine import AlertEngine


# Error patterns per affected service
_DATABASE_ERRORS = [
    {"level": "WARN", "message": "max_connections (100) reached"},
    {"level": "WARN", "message": "connection pool utilization: 100%"},
    {"level": "INFO", "message": "connection queued — waiting for available slot"},
]

_ORDER_SERVICE_ERRORS = [
    {"level": "ERROR", "message": "Connection refused: database:5432"},
    {"level": "ERROR", "message": "Connection pool exhausted — cannot acquire connection"},
    {"level": "ERROR", "message": "Timeout waiting for database connection (30s)"},
    {"level": "ERROR", "message": "Order processing failed: upstream dependency unavailable"},
    {"level": "WARN", "message": "Falling back to read replica — primary unavailable"},
]

_GATEWAY_ERRORS = [
    {"level": "ERROR", "message": "upstream timeout: order-service:8080"},
    {"level": "ERROR", "message": "upstream returned 503: order-service"},
    {"level": "WARN", "message": "5xx error rate elevated: 52%"},
]

# Red herring: cache shows problems that LOOK like a local issue
_CACHE_ERRORS = [
    {"level": "ERROR", "message": "connection timeout to backend — retrying"},
    {"level": "WARN", "message": "cache miss rate elevated: 24%"},
    {"level": "ERROR", "message": "failed to refresh cache entry: upstream error"},
    {"level": "ERROR", "message": "eviction rate: 45% (critical threshold 40%) — possible memory pressure"},
    {"level": "WARN", "message": "cache hit ratio dropped to 0.62 (SLA threshold: 0.80)"},
]

# Red herring: inventory-service shows timeouts too
_INVENTORY_ERRORS = [
    {"level": "ERROR", "message": "stock check failed: order-service timeout"},
    {"level": "WARN", "message": "inventory sync delayed — dependency error"},
]


class MediumCascade(ScenarioBase):
    """Cascading failure: database pool exhaustion -> order-service -> api-gateway.

    The trap: CRITICAL alerts fire on api-gateway and order-service (symptoms).
    Cache and inventory-service also show errors (downstream effects).
    Database only shows WARN-level alerts and its logs are subtle (WARN not ERROR).
    The agent must trace upstream through the dependency chain to find database
    as the root cause, ignoring the louder downstream symptoms.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)
        self._log_gen = LogGenerator(seed=seed)
        self._metrics = MetricsStore(seed=seed)
        self._alerts = AlertEngine()

        # Anomalous metrics — database looks less alarming than downstream
        self._metrics.set_metrics("database", {
            "connections": 100,
            "latency": 800.0,
            "error_rate": 5.0,
            "cpu": 65.0,
        })
        self._metrics.set_metrics("order-service", {
            "error_rate": 60.0,
            "latency": 5000.0,
            "cpu": 45.0,
        })
        self._metrics.set_metrics("api-gateway", {
            "error_rate": 55.0,
            "latency": 5500.0,
        })
        self._metrics.set_metrics("cache", {
            "error_rate": 18.0,
            "latency": 600.0,
        })
        self._metrics.set_metrics("inventory-service", {
            "error_rate": 25.0,
            "latency": 3000.0,
        })

        # Alerts — downstream services are LOUDER than the root cause
        self._alerts.create_alert(
            "CRITICAL", "api-gateway",
            "503 error rate >50% — service degraded", "error_rate_critical",
        )
        self._alerts.create_alert(
            "CRITICAL", "order-service",
            "error rate >60% — multiple upstream failures", "order_error_critical",
        )
        self._alerts.create_alert(
            "HIGH", "inventory-service",
            "stock check failures — dependency timeout", "inventory_failures",
        )
        self._alerts.create_alert(
            "CRITICAL", "cache",
            "cache eviction rate critical — possible memory pressure, consider scale_up",
            "cache_eviction_critical",
        )
        self._alerts.create_alert(
            "WARN", "database",
            "active connections at limit (100/100)", "connection_pool_warn",
        )

    def get_ground_truth(self) -> GroundTruth:
        return GroundTruth(
            severity="P1",
            root_cause_service="database",
            root_cause_category="connection_pool_exhaustion",
            remediation_action="increase_pool_size",
            remediation_target="database",
            affected_services=["database", "order-service", "api-gateway"],
            causal_chain=["database", "order-service", "api-gateway"],
        )

    def get_alerts(self) -> List[Dict[str, Any]]:
        return self._alerts.get_alerts()

    def get_logs(self, service: str, lines: int) -> List[str]:
        if service == "database":
            # Subtle — only WARN level, easy to overlook
            return self._log_gen.generate_logs(service, lines, _DATABASE_ERRORS, error_ratio=0.15)
        if service == "order-service":
            return self._log_gen.generate_logs(service, lines, _ORDER_SERVICE_ERRORS, error_ratio=0.35)
        if service == "api-gateway":
            return self._log_gen.generate_logs(service, lines, _GATEWAY_ERRORS, error_ratio=0.3)
        if service == "cache":
            return self._log_gen.generate_logs(service, lines, _CACHE_ERRORS, error_ratio=0.2)
        if service == "inventory-service":
            return self._log_gen.generate_logs(service, lines, _INVENTORY_ERRORS, error_ratio=0.15)
        return self._log_gen.generate_normal_logs(service, lines)

    def get_metrics(self, service: str, metric: str) -> Dict[str, Any]:
        return self._metrics.get_metrics(service, metric)

    def get_initial_observation_text(self) -> str:
        return (
            "INCIDENT ALERT: Widespread errors across the platform. "
            "api-gateway and order-service reporting critical error rates. "
            "Cache and inventory-service also affected. Multiple services "
            "showing elevated latency and failures. Investigate to find "
            "the root cause."
        )

"""Task 3: Intermittent Auth Failure + Red Herrings — clock skew on auth-service."""
from typing import Any, Dict, List, Optional

from .base import ScenarioBase, GroundTruth
from ..simulation.log_generator import LogGenerator
from ..simulation.metrics_store import MetricsStore
from ..simulation.alert_engine import AlertEngine


# Auth-service errors — the REAL issue but subtle and rare
_AUTH_SERVICE_ERRORS = [
    {"level": "WARN", "message": "JWT validation failed: token not yet valid (clock skew detected)"},
    {"level": "WARN", "message": "System clock offset: +3.2s from NTP server"},
]

# order-service RED HERRING — alarming ERROR-level logs that look like a real crisis
_ORDER_SERVICE_RED_HERRING = [
    {"level": "ERROR", "message": "CPU usage critical: 92% — possible runaway process detected"},
    {"level": "ERROR", "message": "Thread pool exhausted — 48/50 threads busy with batch operations"},
    {"level": "ERROR", "message": "Response time degraded: p99=2400ms (threshold: 500ms)"},
    {"level": "WARN", "message": "Memory pressure: GC running every 200ms during batch processing"},
    {"level": "ERROR", "message": "Request queue backing up — 23 pending requests"},
    {"level": "INFO", "message": "Starting scheduled batch job: nightly_inventory_sync"},
    {"level": "INFO", "message": "Batch processing: 50000 records (this is a scheduled operation)"},
]

# api-gateway — mix of symptoms, not all clearly auth-related
_GATEWAY_ERRORS = [
    {"level": "WARN", "message": "elevated error rate: 12% (mixed 401/503)"},
    {"level": "WARN", "message": "upstream latency spike from multiple services"},
    {"level": "ERROR", "message": "request failed: intermittent upstream errors"},
]

# payment-service — also affected, adds noise
_PAYMENT_ERRORS = [
    {"level": "ERROR", "message": "transaction failed: upstream dependency timeout"},
    {"level": "WARN", "message": "retry limit reached for payment authorization"},
    {"level": "ERROR", "message": "payment gateway returned 500 — possible auth issue"},
]

# inventory-service — mild noise
_INVENTORY_ERRORS = [
    {"level": "WARN", "message": "stock sync delayed — dependency timeout"},
]


class HardIntermittent(ScenarioBase):
    """Intermittent auth failures from clock skew + convincing red herring CPU spike.

    The trap: order-service CPU at 92% with ERROR-level logs about thread
    exhaustion and response degradation looks like the obvious culprit.
    The real issue is auth-service clock skew — but the clock skew logs are
    rare (only ~8% of auth-service lines) and buried among normal auth traffic.
    Multiple misleading alerts across services add confusion.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)
        self._log_gen = LogGenerator(seed=seed)
        self._metrics = MetricsStore(seed=seed)
        self._alerts = AlertEngine()

        # order-service: CPU high, error_rate also elevated (red herring looks real)
        self._metrics.set_metrics("order-service", {
            "cpu": 92.0,
            "memory": 78.0,
            "error_rate": 8.0,
            "latency": 2400.0,
        })
        # auth-service: looks mostly healthy! Only subtle error_rate bump
        self._metrics.set_metrics("auth-service", {
            "error_rate": 6.0,
        })
        # api-gateway: elevated but not extreme
        self._metrics.set_metrics("api-gateway", {
            "error_rate": 12.0,
        })
        # payment-service: also affected
        self._metrics.set_metrics("payment-service", {
            "error_rate": 5.0,
            "latency": 800.0,
        })

        # 8 alerts — order-service is LOUDEST (red herring), auth-service is subtle
        self._alerts.create_alert(
            "CRITICAL", "order-service",
            "CPU usage >90% — possible runaway process", "cpu_critical",
        )
        self._alerts.create_alert(
            "HIGH", "order-service",
            "response time degraded: p99 >2s", "latency_high",
        )
        self._alerts.create_alert(
            "HIGH", "payment-service",
            "transaction failure rate elevated", "payment_failures",
        )
        self._alerts.create_alert(
            "WARN", "api-gateway",
            "mixed error rate elevated (401s + 503s)", "gateway_errors",
        )
        self._alerts.create_alert(
            "WARN", "auth-service",
            "intermittent 401 responses (low volume)", "auth_intermittent",
        )
        self._alerts.create_alert(
            "WARN", "inventory-service",
            "stock sync delayed", "inventory_sync_delay",
        )
        self._alerts.create_alert(
            "INFO", "cache",
            "eviction rate elevated", "cache_eviction",
        )
        self._alerts.create_alert(
            "INFO", "database",
            "slow query detected: 450ms", "slow_query",
        )

    def get_ground_truth(self) -> GroundTruth:
        return GroundTruth(
            severity="P2",
            root_cause_service="auth-service",
            root_cause_category="clock_skew",
            remediation_action="config_change",
            remediation_target="auth-service",
            affected_services=["auth-service", "api-gateway"],
            causal_chain=["auth-service", "api-gateway"],
        )

    def get_alerts(self) -> List[Dict[str, Any]]:
        return self._alerts.get_alerts()

    def get_logs(self, service: str, lines: int) -> List[str]:
        if service == "auth-service":
            # Very low error ratio — clock skew errors are rare and buried
            return self._log_gen.generate_logs(service, lines, _AUTH_SERVICE_ERRORS, error_ratio=0.08)
        if service == "order-service":
            # High error ratio — red herring looks very alarming
            return self._log_gen.generate_logs(service, lines, _ORDER_SERVICE_RED_HERRING, error_ratio=0.45)
        if service == "api-gateway":
            return self._log_gen.generate_logs(service, lines, _GATEWAY_ERRORS, error_ratio=0.15)
        if service == "payment-service":
            return self._log_gen.generate_logs(service, lines, _PAYMENT_ERRORS, error_ratio=0.2)
        if service == "inventory-service":
            return self._log_gen.generate_logs(service, lines, _INVENTORY_ERRORS, error_ratio=0.1)
        return self._log_gen.generate_normal_logs(service, lines)

    def get_metrics(self, service: str, metric: str) -> Dict[str, Any]:
        return self._metrics.get_metrics(service, metric)

    def get_initial_observation_text(self) -> str:
        return (
            "INCIDENT ALERT: Multiple services degraded. order-service CPU critical "
            "at 92% with response time spikes. Payment failures increasing. "
            "Intermittent errors across api-gateway. Investigate the most critical "
            "alerts first."
        )

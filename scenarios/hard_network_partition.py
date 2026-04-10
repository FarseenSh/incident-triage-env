"""Task 5: Network Partition — split-brain between services, subtle and misleading."""
from typing import Any, Dict, List, Optional

from .base import ScenarioBase, GroundTruth
from ..simulation.log_generator import LogGenerator
from ..simulation.metrics_store import MetricsStore
from ..simulation.alert_engine import AlertEngine


# payment-service is the root cause — network partition isolates it
_PAYMENT_ERRORS = [
    {"level": "ERROR", "message": "connection timeout: message-queue:9092 (unreachable)"},
    {"level": "ERROR", "message": "TCP connection reset by peer: message-queue"},
    {"level": "WARN", "message": "network interface eth0: packet loss 85%"},
    {"level": "ERROR", "message": "DNS resolution failed for message-queue.internal"},
]

# message-queue appears healthy from its own perspective (it IS healthy)
_MQ_ERRORS = [
    {"level": "WARN", "message": "consumer payment-service disconnected — no heartbeat"},
    {"level": "INFO", "message": "queue depth increasing: payments=247 (no consumers)"},
]

# order-service sees payment timeouts (downstream of partition)
_ORDER_SERVICE_ERRORS = [
    {"level": "ERROR", "message": "payment authorization timeout — payment-service:443 unreachable"},
    {"level": "ERROR", "message": "order stuck in PENDING — payment confirmation never received"},
    {"level": "WARN", "message": "circuit breaker HALF-OPEN for payment-service"},
]

# Red herring: database shows high latency (unrelated — vacuum is running)
_DATABASE_ERRORS = [
    {"level": "WARN", "message": "autovacuum running on large table: orders (42GB)"},
    {"level": "WARN", "message": "query latency elevated during vacuum: avg 340ms"},
    {"level": "INFO", "message": "vacuum progress: 67% complete on orders table"},
]

# api-gateway sees mixed errors
_GATEWAY_ERRORS = [
    {"level": "ERROR", "message": "POST /api/v1/orders/pay returning 504"},
    {"level": "WARN", "message": "partial service degradation: payments path affected"},
]

# auth-service: red herring — token refresh surge (normal during incident)
_AUTH_ERRORS = [
    {"level": "WARN", "message": "token refresh rate elevated: 340 req/s (normal: 120 req/s)"},
    {"level": "INFO", "message": "all authentications succeeding — no auth failures"},
]


class HardNetworkPartition(ScenarioBase):
    """Network partition isolates payment-service from message-queue.

    The trap: Multiple red herrings compete for attention.
    - database shows high latency (but it's just autovacuum — normal)
    - auth-service has elevated token refreshes (but all succeeding)
    - order-service has the loudest errors (but it's downstream)
    - message-queue appears healthy from its own logs

    The real issue: payment-service can't reach message-queue due to
    network partition. Only payment-service logs show the TCP/DNS errors
    that reveal the network issue. The agent must notice this is a
    network_partition (not a service crash) and apply failover.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)
        self._log_gen = LogGenerator(seed=seed)
        self._metrics = MetricsStore(seed=seed)
        self._alerts = AlertEngine()

        # payment-service: high error rate, normal CPU/memory (not a crash)
        self._metrics.set_metrics("payment-service", {
            "error_rate": 70.0,
            "latency": 30000.0,
            "cpu": 25.0,
            "memory": 45.0,
        })
        # order-service: elevated errors from payment path
        self._metrics.set_metrics("order-service", {
            "error_rate": 35.0,
            "latency": 3500.0,
        })
        # database: latency AND error_rate elevated — convincing red herring
        self._metrics.set_metrics("database", {
            "latency": 340.0,
            "error_rate": 18.0,
            "cpu": 80.0,
        })
        # auth-service: small errors to correlate with refresh spike
        self._metrics.set_metrics("auth-service", {
            "latency": 45.0,
            "error_rate": 2.5,
        })
        self._metrics.set_metrics("api-gateway", {
            "error_rate": 20.0,
        })

        # 8 alerts — real cause buried among noise
        self._alerts.create_alert(
            "CRITICAL", "order-service",
            "payment processing failures — orders stuck in PENDING", "order_payment_stuck",
        )
        self._alerts.create_alert(
            "HIGH", "database",
            "query latency elevated (autovacuum running)", "db_latency_high",
        )
        self._alerts.create_alert(
            "HIGH", "payment-service",
            "connection failures to downstream services", "payment_conn_failures",
        )
        self._alerts.create_alert(
            "HIGH", "api-gateway",
            "payment endpoints returning 504", "gateway_payment_timeout",
        )
        self._alerts.create_alert(
            "WARN", "auth-service",
            "token refresh rate spike (3x normal)", "auth_refresh_spike",
        )
        self._alerts.create_alert(
            "WARN", "message-queue",
            "consumer disconnected: payment-service", "mq_consumer_lost",
        )
        self._alerts.create_alert(
            "INFO", "cache",
            "eviction rate slightly elevated", "cache_eviction",
        )
        self._alerts.create_alert(
            "INFO", "inventory-service",
            "stock reservation timeout for pending orders", "inventory_timeout",
        )

    def get_ground_truth(self) -> GroundTruth:
        return GroundTruth(
            severity="P1",
            root_cause_service="payment-service",
            root_cause_category="network_partition",
            remediation_action="failover",
            remediation_target="payment-service",
            affected_services=["payment-service", "order-service", "api-gateway"],
            causal_chain=["payment-service", "order-service", "api-gateway"],
        )

    def get_alerts(self) -> List[Dict[str, Any]]:
        return self._alerts.get_alerts()

    def get_logs(self, service: str, lines: int) -> List[str]:
        if service == "payment-service":
            return self._log_gen.generate_logs(service, lines, _PAYMENT_ERRORS, error_ratio=0.35)
        if service == "message-queue":
            return self._log_gen.generate_logs(service, lines, _MQ_ERRORS, error_ratio=0.1)
        if service == "order-service":
            return self._log_gen.generate_logs(service, lines, _ORDER_SERVICE_ERRORS, error_ratio=0.3)
        if service == "database":
            # Red herring — looks alarming but it's just vacuum
            return self._log_gen.generate_logs(service, lines, _DATABASE_ERRORS, error_ratio=0.25)
        if service == "api-gateway":
            return self._log_gen.generate_logs(service, lines, _GATEWAY_ERRORS, error_ratio=0.15)
        if service == "auth-service":
            return self._log_gen.generate_logs(service, lines, _AUTH_ERRORS, error_ratio=0.1)
        return self._log_gen.generate_normal_logs(service, lines)

    def get_metrics(self, service: str, metric: str) -> Dict[str, Any]:
        return self._metrics.get_metrics(service, metric)

    def get_initial_observation_text(self) -> str:
        return (
            "INCIDENT ALERT: Payment processing completely down. Orders stuck "
            "in PENDING state. Database showing elevated latency. Multiple "
            "services reporting errors. Auth token refresh rate spiking. "
            "Investigate all affected services."
        )

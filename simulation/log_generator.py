"""Generate realistic timestamped log lines for microservices."""
import random
from typing import Dict, List, Optional


# Normal log messages per service type
_NOISE_MESSAGES: Dict[str, List[str]] = {
    "api-gateway": [
        "Request received: GET /api/v1/orders",
        "Request received: POST /api/v1/orders",
        "Request received: GET /api/v1/health",
        "Rate limiter check passed",
        "Routing request to upstream service",
        "Response sent: 200 OK (42ms)",
        "Response sent: 200 OK (38ms)",
        "TLS handshake completed",
        "Connection pool stats: active=12, idle=8",
    ],
    "auth-service": [
        "JWT token validated successfully",
        "User authenticated: uid=a]f8c2e1",
        "Session refreshed for uid=b7d3a4f2",
        "Token cache hit ratio: 94.2%",
        "LDAP sync completed: 0 changes",
        "Password policy check passed",
        "OAuth2 token issued",
    ],
    "order-service": [
        "Processing order #ORD-29841",
        "Order #ORD-29840 completed successfully",
        "Inventory check passed for SKU-1234",
        "Payment authorization requested",
        "Database query executed in 12ms",
        "Cache hit for product catalog",
        "Order validation passed",
    ],
    "inventory-service": [
        "Stock level check: SKU-1234 qty=142",
        "Reservation created for order #ORD-29841",
        "Inventory sync completed",
        "Warehouse API response: 200 (15ms)",
        "Stock alert threshold check passed",
    ],
    "payment-service": [
        "Payment processed: txn=TXN-88421",
        "Stripe webhook received",
        "Refund initiated: txn=TXN-88400",
        "Payment gateway health: OK",
        "PCI compliance check passed",
    ],
    "database": [
        "Query executed: SELECT * FROM orders (3ms)",
        "Connection pool stats: active=24/100, idle=76",
        "Slow query log: 0 entries",
        "Checkpoint completed",
        "WAL segment archived",
        "Vacuum completed on table: orders",
        "Replication lag: 0ms",
    ],
    "cache": [
        "Cache hit ratio: 96.3%",
        "Key eviction: 0 keys evicted",
        "Memory usage: 1.2GB / 4GB",
        "Connected clients: 18",
        "RDB snapshot completed",
    ],
    "message-queue": [
        "Message published to queue: payments",
        "Consumer group lag: 0 messages",
        "Queue depth: payments=3, notifications=1",
        "Health check: all brokers healthy",
        "Partition rebalance completed",
    ],
}

_WARN_MESSAGES = [
    "GC pause: 45ms (within threshold)",
    "Request latency above P95 threshold",
    "Retry attempt 1/3 for upstream call",
    "Connection timeout, retrying",
    "High memory watermark approaching",
]


class LogGenerator:
    """Generates realistic log lines with optional error injection."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._base_ts = "2024-01-15T14:00:00"

    def _make_timestamp(self, offset_seconds: float) -> str:
        """Generate an ISO 8601 timestamp offset from the base time."""
        total_seconds = int(offset_seconds)
        millis = int((offset_seconds - total_seconds) * 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"2024-01-15T{14 + hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}Z"

    def _make_trace_id(self) -> str:
        return f"trace-{self._rng.randint(0, 0xFFFFFFFF):08x}"

    def generate_normal_logs(self, service: str, lines: int) -> List[str]:
        """Generate healthy noise logs for a service (no errors)."""
        return self.generate_logs(service, lines, error_patterns=None)

    def generate_logs(
        self,
        service: str,
        lines: int,
        error_patterns: Optional[List[Dict[str, str]]] = None,
        error_ratio: float = 0.3,
    ) -> List[str]:
        """Generate log lines for a service.

        Args:
            service: Service name.
            lines: Number of log lines to generate.
            error_patterns: Optional list of dicts with "level" and "message" keys
                            to inject among normal logs.
            error_ratio: Fraction of lines that should be error patterns (0.0-1.0).
                         Higher = more errors visible. Default 0.3.
        Returns:
            List of formatted log strings.
        """
        result = []
        noise_msgs = _NOISE_MESSAGES.get(service, ["Processing request"])
        errors = error_patterns or []

        # Decide positions where error lines will appear
        error_positions = set()
        if errors:
            n_errors = max(1, int(lines * error_ratio))
            for i in range(n_errors):
                pos = self._rng.randint(lines // 4, lines - 1)
                error_positions.add(pos)

        error_idx = 0
        for i in range(lines):
            offset = i * self._rng.uniform(0.5, 3.0)
            ts = self._make_timestamp(offset)
            trace_id = self._make_trace_id()

            if i in error_positions and errors:
                err = errors[error_idx % len(errors)]
                error_idx += 1
                level = err.get("level", "ERROR")
                msg = err.get("message", "Unknown error")
            elif self._rng.random() < 0.15:
                level = "WARN"
                msg = self._rng.choice(_WARN_MESSAGES)
            else:
                level = "INFO"
                msg = self._rng.choice(noise_msgs)

            result.append(f"{ts} [{level}] {service} [{trace_id}] {msg}")

        # Sort chronologically — real logs are time-ordered
        result.sort()
        return result

"""Simulated per-service metrics with healthy defaults and scenario overrides."""
import random
from typing import Any, Dict, Optional


# Healthy metric ranges
_HEALTHY_RANGES = {
    "cpu_percent": (15.0, 35.0),
    "memory_percent": (40.0, 60.0),
    "error_rate_percent": (0.0, 2.0),
    "latency_p99_ms": (50.0, 200.0),
    "connections_active": (10, 30),
}

# Map short metric names to full keys
_METRIC_ALIASES = {
    "cpu": "cpu_percent",
    "memory": "memory_percent",
    "error_rate": "error_rate_percent",
    "latency": "latency_p99_ms",
    "connections": "connections_active",
}


class MetricsStore:
    """Per-service metrics with healthy defaults and scenario overrides."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._overrides: Dict[str, Dict[str, Any]] = {}
        # Pre-generate healthy baselines so they're stable across calls
        self._baselines: Dict[str, Dict[str, Any]] = {}

    def _get_baseline(self, service: str) -> Dict[str, Any]:
        """Return stable healthy baseline metrics for a service."""
        if service not in self._baselines:
            self._baselines[service] = {
                "cpu_percent": round(self._rng.uniform(*_HEALTHY_RANGES["cpu_percent"]), 1),
                "memory_percent": round(self._rng.uniform(*_HEALTHY_RANGES["memory_percent"]), 1),
                "error_rate_percent": round(self._rng.uniform(*_HEALTHY_RANGES["error_rate_percent"]), 2),
                "latency_p99_ms": round(self._rng.uniform(*_HEALTHY_RANGES["latency_p99_ms"]), 0),
                "connections_active": self._rng.randint(*_HEALTHY_RANGES["connections_active"]),
            }
        return self._baselines[service]

    def set_metrics(self, service: str, overrides: Dict[str, Any]) -> None:
        """Set anomalous metric values for a service.

        Args:
            service: Service name.
            overrides: Dict of metric_name -> value. Supports both full names
                       (cpu_percent) and short aliases (cpu).
        """
        resolved = {}
        for key, value in overrides.items():
            full_key = _METRIC_ALIASES.get(key, key)
            resolved[full_key] = value
        self._overrides[service] = resolved

    def get_metrics(self, service: str, metric: str = "all") -> Dict[str, Any]:
        """Return metrics for a service.

        Args:
            service: Service name.
            metric: "all" for all metrics, or a specific name (cpu, memory, etc.)
        Returns:
            Dict of metric values.
        """
        baseline = dict(self._get_baseline(service))
        # Apply overrides
        if service in self._overrides:
            baseline.update(self._overrides[service])

        if metric == "all":
            return baseline

        # Resolve alias
        full_key = _METRIC_ALIASES.get(metric, metric)
        if full_key in baseline:
            return {full_key: baseline[full_key]}
        return {"error": f"Unknown metric: {metric}"}

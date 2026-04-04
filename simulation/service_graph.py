"""Microservice topology graph with dependency edges."""
from typing import Dict, List


# Hardcoded edges from spec section 3
_EDGES = [
    ("api-gateway", "auth-service"),
    ("api-gateway", "order-service"),
    ("order-service", "inventory-service"),
    ("order-service", "payment-service"),
    ("order-service", "database"),
    ("order-service", "cache"),
    ("auth-service", "database"),
    ("payment-service", "message-queue"),
]

_ALL_SERVICES = [
    "api-gateway",
    "auth-service",
    "order-service",
    "inventory-service",
    "payment-service",
    "database",
    "cache",
    "message-queue",
]


class ServiceGraph:
    """Directed dependency graph for the microservice topology."""

    def __init__(self):
        self._dependencies: Dict[str, List[str]] = {s: [] for s in _ALL_SERVICES}
        self._dependents: Dict[str, List[str]] = {s: [] for s in _ALL_SERVICES}
        for src, dst in _EDGES:
            self._dependencies[src].append(dst)
            self._dependents[dst].append(src)

    def get_dependencies(self, service: str) -> List[str]:
        """Return downstream services that *service* depends on."""
        return list(self._dependencies.get(service, []))

    def get_dependents(self, service: str) -> List[str]:
        """Return upstream services that depend on *service*."""
        return list(self._dependents.get(service, []))

    def get_topology_description(self) -> str:
        """Return a human-readable description of the service dependency graph."""
        lines = ["Service Dependency Graph:"]
        for service in _ALL_SERVICES:
            deps = self._dependencies[service]
            if deps:
                lines.append(f"  {service} -> {', '.join(deps)}")
            else:
                lines.append(f"  {service} (no outgoing dependencies)")
        return "\n".join(lines)

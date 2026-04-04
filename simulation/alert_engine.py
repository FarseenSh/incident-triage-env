"""Alert engine that stores and returns alerts sorted by severity."""
from typing import Any, Dict, List

# Severity ordering for sorting (most critical first)
_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "WARN": 2, "INFO": 3}


class AlertEngine:
    """Container for scenario-defined alerts."""

    def __init__(self):
        self._alerts: List[Dict[str, Any]] = []
        self._alert_counter = 0

    def create_alert(
        self,
        severity: str,
        service: str,
        message: str,
        rule_name: str,
        timestamp: str = "",
    ) -> Dict[str, Any]:
        """Create and store an alert, returning the alert dict.

        Args:
            severity: CRITICAL, HIGH, WARN, or INFO.
            service: Service that triggered the alert.
            message: Alert description.
            rule_name: Name of the alerting rule.
            timestamp: Optional ISO 8601 timestamp. Auto-generated if empty.
        Returns:
            The alert dict that was added.
        """
        if not timestamp:
            minutes = self._alert_counter * 2
            timestamp = f"2024-01-15T14:{minutes:02d}:00Z"
        self._alert_counter += 1
        alert = {
            "severity": severity,
            "service": service,
            "message": message,
            "timestamp": timestamp,
            "rule_name": rule_name,
        }
        self._alerts.append(alert)
        return alert

    def add_alert(
        self,
        severity: str,
        service: str,
        message: str,
        rule_name: str,
        timestamp: str = "",
    ) -> None:
        """Add an alert (convenience alias for create_alert)."""
        self.create_alert(severity, service, message, rule_name, timestamp)

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Return all alerts sorted by severity (CRITICAL first)."""
        return sorted(
            self._alerts,
            key=lambda a: _SEVERITY_ORDER.get(a["severity"], 99),
        )

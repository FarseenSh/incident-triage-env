"""FastAPI server for the Incident Triage environment."""
import json
from typing import Any, Dict

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from pydantic import field_validator

from .incident_environment import IncidentTriageEnvironment


class TriageCallToolAction(CallToolAction):
    """CallToolAction that accepts JSON strings for arguments.
    The web UI and some clients send arguments as JSON strings instead of dicts.
    Without this validator, those requests crash with a validation error.
    Pattern copied from finqa_env/server/app.py.
    """
    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, v: Any) -> Dict[str, Any]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {"raw_input": v}
        return v


def _env_factory():
    """Create a new IncidentTriageEnvironment instance for each session."""
    return IncidentTriageEnvironment()


app = create_app(
    _env_factory,
    CallToolAction,         # Must be CallToolAction (not subclass) for MCP deserialization
    CallToolObservation,
    env_name="incident_triage_env",
    max_concurrent_envs=4,
)


def main():
    """Entry point for `uv run server` or `python -m server.app`."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

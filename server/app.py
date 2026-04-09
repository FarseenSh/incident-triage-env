"""FastAPI server for the Incident Triage environment."""
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from .incident_environment import IncidentTriageEnvironment


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

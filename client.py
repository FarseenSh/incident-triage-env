"""
Client for the Incident Response Triage environment.
Connects to a running server and provides MCP tool-calling interface.

Example:
    async with IncidentTriageEnv(base_url="http://localhost:8000") as env:
        await env.reset(task_name="easy_oom_crash")
        tools = await env.list_tools()
        result = await env.call_tool("get_alerts")
        result = await env.call_tool("read_logs", service="order-service", lines=50)
        result = await env.call_tool("diagnose",
            root_cause_service="order-service",
            root_cause_category="memory_exhaustion")
        result = await env.call_tool("remediate",
            action="restart_service",
            target_service="order-service")
        result = await env.call_tool("submit_report")
"""
from openenv.core.mcp_client import MCPToolClient


class IncidentTriageEnv(MCPToolClient):
    """MCP client for the Incident Response Triage environment."""
    pass  # MCPToolClient provides all needed functionality

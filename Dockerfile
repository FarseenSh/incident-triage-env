# Dockerfile — Incident Triage Environment (rebuilt 2026-04-11)
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder
WORKDIR /app

# Copy project — cache-bust: 2026-04-11T04:40
COPY . /app/incident_triage_env
WORKDIR /app/incident_triage_env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; fi

# Git needed for some pip installs
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install deps then project in editable mode so source files are used at runtime
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then uv sync --frozen --no-install-project; \
    else uv sync --no-install-project; fi
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then uv sync --frozen; \
    else uv sync; fi

# Final runtime stage
FROM ${BASE_IMAGE}
WORKDIR /app
COPY --from=builder /app/incident_triage_env/.venv /app/.venv
COPY --from=builder /app/incident_triage_env /app/incident_triage_env
# Strip YAML frontmatter from README for the OpenEnv web interface
RUN sed '1{/^---$/!q;};1,/^---$/d' /app/incident_triage_env/README.md > /app/README.md
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Enable web interface for HF Spaces (openenv push does this automatically,
# but we set it explicitly so manual docker build also works)
ENV ENABLE_WEB_INTERFACE=true

# Healthcheck using Python (no curl dependency needed)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "incident_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
# Rebuild trigger 1744319100

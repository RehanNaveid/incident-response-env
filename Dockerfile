FROM python:3.12-slim

WORKDIR /app

# Install minimal system deps
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install uv (fast + reproducible)
RUN pip install --no-cache-dir uv

# Copy project
COPY . .

# Install dependencies from pyproject + uv.lock
RUN uv sync

# Expose port (HF uses dynamic PORT)
EXPOSE 7860

# Healthcheck (important for HF + validator)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:7860/health || exit 1

# Run server via uv (uses [project.scripts])
CMD ["uv", "run", "server"]
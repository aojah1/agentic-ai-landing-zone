# Dockerfile
FROM --platform=linux/amd64 python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MCP_SSE_HOST=0.0.0.0 \
    MCP_SSE_PORT=8001

WORKDIR /app

# 1) Install build deps (for any packages needing compilation), upgrade pip
# 2) Install your deps from the *root* requirements.txt
# 3) Remove build deps to keep the image lean
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential gcc libffi-dev libssl-dev cargo && \
    python3.13 -m pip install --upgrade pip && \
    python3.13 -m pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential gcc cargo && \
    rm -rf /var/lib/apt/lists/*

# Copy your app
COPY . .

EXPOSE 8001

# Your main.py already calls uvicorn.run(app, host=MCP_SSE_HOST, port=MCP_SSE_PORT)
CMD ["python3.13", "-m", "src.main"]

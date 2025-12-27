# ==================================================
# V3 Climate Forecaster - Production Dockerfile
# ==================================================
# Multi-stage build for minimal image size
# Runs V3 Climate-Aware Transformer API on port 8000

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-v3.txt .
RUN pip install --no-cache-dir --user -r requirements-v3.txt

# ==================================================
# Production Stage
# ==================================================
FROM python:3.11-slim AS production

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy V3 application code
COPY v3/src ./v3/src
COPY v3/app ./v3/app
COPY v3/models ./v3/models

# Set environment
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "v3.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ==================================================
# Development Stage (for local testing with hot reload)
# ==================================================
FROM production AS development

USER root

# Install dev tools
RUN pip install --no-cache-dir pytest httpx

USER appuser

# Development mode with auto-reload
CMD ["python", "-m", "uvicorn", "v3.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

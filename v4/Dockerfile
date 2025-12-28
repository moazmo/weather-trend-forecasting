# ==================================================
# V4 Ensemble Forecaster - Production Dockerfile
# ==================================================
# Multi-stage build for minimal image size
# Runs V4 Ensemble Weather Forecaster API on port 8002

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user xgboost

# ==================================================
# Production Stage
# ==================================================
FROM python:3.11-slim AS production

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy V4 application code
COPY v4/src ./v4/src
COPY v4/app ./v4/app
COPY v4/models ./v4/models

# Copy shared dependencies if needed
COPY v2/models/location_stats.csv ./v2/models/location_stats.csv

# Set environment
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "v4.app.main:app", "--host", "0.0.0.0", "--port", "8002"]

# ==================================================
# Development Stage (for local testing with hot reload)
# ==================================================
FROM production AS development

USER root

# Install dev tools
RUN pip install --no-cache-dir pytest httpx

USER appuser

# Development mode with auto-reload
CMD ["python", "-m", "uvicorn", "v4.app.main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]

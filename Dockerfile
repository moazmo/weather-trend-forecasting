# ==================================================
# Weather Trend Forecasting - Production Dockerfile
# ==================================================
# Multi-stage build for minimal image size
# Runs V4 Advanced Transformer API on port 8001

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ==================================================
# Production Stage
# ==================================================
FROM python:3.11-slim AS production

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY v2/app ./v2/app
COPY v2/models ./v2/models
COPY data/processed ./data/processed

# Set environment
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/api/health').raise_for_status()" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "v2.app.main:app", "--host", "0.0.0.0", "--port", "8001"]

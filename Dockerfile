# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependencies first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ src/
COPY models/ models/
COPY data/cache/ data/cache/
# Note: In a real prod environment, we might download models from S3 instead of COPY

# Expose API port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

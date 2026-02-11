FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    transformers==4.36.2 \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    accelerate==0.25.0

# Copy worker script
COPY transformers_worker.py .

# Expose worker port
EXPOSE 8000

# Default command (will be overridden by docker-compose)
ENTRYPOINT ["python", "transformers_worker.py"]
CMD ["--model-path", "/models/default"]

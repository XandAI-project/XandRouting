FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY transformers_worker.py .

# Expose port
EXPOSE 8000

# Set environment variables
ENV ENGINE_TYPE=transformers
ENV PORT=8000

# Run the worker
CMD ["python", "transformers_worker.py"]

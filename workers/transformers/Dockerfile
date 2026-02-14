FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install PyTorch nightly with full CUDA support (includes sm_120 for Blackwell architecture)
RUN pip3 install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# Install other Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy worker code
COPY transformers_worker.py .

# Expose port
EXPOSE 8000

# Set environment variables
ENV ENGINE_TYPE=transformers
ENV PORT=8000

# Run the worker
CMD ["python3", "transformers_worker.py"]

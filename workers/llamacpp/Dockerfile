FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install llama-cpp-python with CUDA support (using new GGML_CUDA flag)
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir llama-cpp-python>=0.3.0

# Install other dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic httpx

# Copy worker code
COPY llamacpp_worker.py .

# Expose port
EXPOSE 8000

# Set environment variables
ENV ENGINE_TYPE=llamacpp
ENV PORT=8000

# Run the worker
CMD ["python", "llamacpp_worker.py"]

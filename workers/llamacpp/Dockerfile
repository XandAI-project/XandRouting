FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DOCKER_ARCH=all

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

# Set up CUDA stub library for build-time linking
# This is required because libcuda.so.1 is only available at runtime with nvidia-docker
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && \
    ldconfig

# Copy requirements
COPY requirements.txt .

# Install llama-cpp-python with CUDA support
# Disable building tools/examples that cause linking issues
# Only build the Python bindings we need
RUN CMAKE_ARGS="-DGGML_CUDA=on -DLLAMA_BUILD_EXAMPLES=off -DLLAMA_BUILD_TESTS=off -DLLAMA_BUILD_TOOLS=off" \
    pip install --no-cache-dir llama-cpp-python>=0.3.0

# Remove CUDA stubs from runtime linker path (only needed for build)
RUN rm /etc/ld.so.conf.d/cuda-stubs.conf && ldconfig

# Install other dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic

# Copy worker code
COPY llamacpp_worker.py .

# Expose port
EXPOSE 8000

# Set environment variables
ENV ENGINE_TYPE=llamacpp
ENV PORT=8000

# Run the worker
CMD ["python", "llamacpp_worker.py"]

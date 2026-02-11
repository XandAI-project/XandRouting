FROM vllm/vllm-openai:v0.15.1

# Set working directory
WORKDIR /app

# Copy worker code
COPY vllm_worker.py .

# Note: Base image already includes all necessary dependencies:
# - vLLM 0.15.1 with CUDA support
# - FastAPI, uvicorn, pydantic
# - torch, transformers<5
# No additional pip installs needed

# Expose port
EXPOSE 8000

# Set environment variables
ENV ENGINE_TYPE=vllm
ENV PORT=8000
ENV CUDA_VISIBLE_DEVICES=0

# Override the vLLM base image entrypoint to run our custom worker
ENTRYPOINT []
CMD ["python3", "vllm_worker.py"]

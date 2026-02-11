FROM vllm/vllm-openai:latest

# Volume for models
VOLUME ["/models"]

# Expose vLLM OpenAI API port
EXPOSE 8000

# Default entrypoint is already set by base image
# Command will be overridden by docker-compose with specific model args
CMD ["--host", "0.0.0.0", "--port", "8000"]

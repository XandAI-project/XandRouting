# Quick Start Guide - Microservices Architecture

## Prerequisites

- Docker with Docker Compose v3.8+
- NVIDIA GPU with CUDA support
- nvidia-docker2 runtime
- At least one model in the `./models/` directory

## Quick Start

### 1. Build All Services

```bash
docker compose build
```

This will build:
- Gateway service (minimal dependencies)
- vLLM engine service
- llama.cpp engine service
- Transformers engine service

**Build Time:** 10-20 minutes (first time)

### 2. Start Services

```bash
docker compose up -d
```

**Startup Time:** 30-60 seconds

### 3. Verify Health

```bash
curl http://localhost:8080/health
```

Expected output:

```json
{
  "status": "healthy",
  "gateway": "healthy",
  "engines": {
    "vllm": {"status": "healthy", ...},
    "llamacpp": {"status": "healthy", ...},
    "transformers": {"status": "healthy", ...}
  }
}
```

### 4. Test Inference

#### Using vLLM Backend

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/YOUR_MODEL_NAME",
    "backend": "vllm",
    "device": "cuda",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

#### Using llama.cpp Backend (GGUF models)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/YOUR_MODEL.gguf",
    "backend": "llamacpp",
    "device": "cuda",
    "n_gpu_layers": -1,
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

#### Using Transformers Backend

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/YOUR_MODEL_NAME",
    "backend": "transformers",
    "device": "cuda",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Common Commands

### View Logs

```bash
# All services
docker compose logs -f

# Gateway only
docker compose logs -f gateway

# Specific engine
docker compose logs -f vllm-engine
docker compose logs -f llamacpp-engine
docker compose logs -f transformers-engine
```

### Check Status

```bash
docker compose ps
```

### Restart Service

```bash
# Restart specific service
docker compose restart gateway

# Restart all
docker compose restart
```

### Stop Services

```bash
docker compose down
```

### Rebuild After Changes

```bash
# Rebuild specific service
docker compose build gateway

# Rebuild all
docker compose build

# Rebuild and restart
docker compose up -d --build
```

## Port Configuration

- **Gateway:** `http://localhost:8080` (exposed to host)
- **Workers:** Internal only (not accessible from host directly)
- **CORS:** Enabled by default (`CORS_ORIGINS=*`)

## Environment Variables

You can customize via environment variables in `docker-compose.yml`:

### Gateway

```yaml
environment:
  - PORT=8080
  - CORS_ORIGINS=*  # Change for production
  - VLLM_SERVICE=vllm-engine:8000
  - LLAMACPP_SERVICE=llamacpp-engine:8000
  - TRANSFORMERS_SERVICE=transformers-engine:8000
```

### Workers

```yaml
environment:
  - PORT=8000
  - CUDA_VISIBLE_DEVICES=0  # GPU device ID
```

## Troubleshooting

### Service Won't Start

Check logs:

```bash
docker compose logs <service-name>
```

### GPU Not Available

Verify GPU access:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Connection Refused

Ensure all services are running:

```bash
docker compose ps
curl http://localhost:8080/health
```

### Model Not Found

Check model path:

```bash
docker compose exec gateway ls -la /models
```

Models should be in `./models/` directory on the host.

### Out of Memory

Reduce GPU memory usage:

```json
{
  "backend": "vllm",
  "gpu_memory_utilization": 0.5
}
```

Or reduce GPU layers:

```json
{
  "backend": "llamacpp",
  "n_gpu_layers": 20
}
```

## API Endpoints

### Core Endpoints

- `GET /` - Gateway status and engine health
- `GET /health` - Health check
- `POST /v1/chat/completions` - Chat completions (main endpoint)

### Model Management

- `GET /v1/models/loaded` - List loaded models
- `GET /v1/models/inventory` - List available models in /models
- `GET /v1/models/stats` - Model statistics
- `POST /v1/models/unload` - Unload specific model
- `POST /v1/models/unload-all` - Unload all models

### Model Downloads

- `POST /v1/models/download` - Download model from HuggingFace
- `GET /v1/models/download` - List download jobs
- `GET /v1/models/download/{job_id}` - Get download status
- `DELETE /v1/models/download/{job_id}` - Cancel download
- `POST /v1/models/verify-repo` - Verify HuggingFace repository

## Next Steps

- Read [MICROSERVICES_ARCHITECTURE.md](MICROSERVICES_ARCHITECTURE.md) for detailed architecture
- Check [README.md](README.md) for API documentation
- See [EXAMPLES.md](EXAMPLES.md) for usage examples
- Configure CORS for production (set `CORS_ORIGINS` to specific domains)
- Set up monitoring with Prometheus/Grafana (future enhancement)

## Support

For issues or questions:
- Check logs: `docker compose logs -f`
- Review [MICROSERVICES_ARCHITECTURE.md](MICROSERVICES_ARCHITECTURE.md)
- Open an issue on GitHub

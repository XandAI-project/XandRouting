---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

<!-- A clear and concise description of the bug -->

## Environment

**Server Information:**
- OS: <!-- e.g., Ubuntu 22.04, Windows 11 -->
- Docker Version: <!-- e.g., 24.0.5 -->
- Docker Compose Version: <!-- e.g., 2.20.0 -->
- GPU: <!-- e.g., NVIDIA RTX 4090, None (CPU only) -->
- CUDA Version: <!-- e.g., 12.1, N/A -->
- NVIDIA Driver Version: <!-- e.g., 535.86.10, N/A -->

**Container Information:**
- Gateway Container Status: <!-- running/stopped/error -->
- Gateway Image Tag: <!-- e.g., latest, v2.0.0 -->

## Steps to Reproduce

<!-- Provide detailed steps to reproduce the bug -->

1. Step 1
2. Step 2
3. Step 3
4. ...

## Expected Behavior

<!-- What you expected to happen -->

## Actual Behavior

<!-- What actually happened -->

## Request Details

**API Request:**
```bash
# Paste the exact curl command or code that triggered the bug
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/...",
    "backend": "...",
    "device": "...",
    ...
  }'
```

**Response:**
```json
{
  "error": "..."
}
```

## Logs

**Gateway Container Logs:**
```
# Run: docker compose logs gateway
# Paste relevant logs here
```

**Error Stack Trace (if applicable):**
```
# Paste full error stack trace
```

## Model Information

**Model Details:**
- Model Name: <!-- e.g., Qwen2.5-Coder-7B-Instruct-GGUF -->
- Model Size: <!-- e.g., 7B parameters, 4.2GB on disk -->
- Quantization: <!-- e.g., Q4_K_M, FP16, None -->
- Download Method: <!-- API download, manual, pre-existing -->

**Model Directory Structure:**
```bash
# Run: ls -lh /models/your-model/
# Paste output here
```

## Configuration

**Environment Variables:**
```bash
DEFAULT_TTL=...
CLEANUP_INTERVAL=...
# Add any custom environment variables
```

**Docker Compose Configuration:**
```yaml
# Paste relevant sections from docker-compose.yml if modified
```

## Additional Context

<!-- Any additional information that might be helpful -->
- Screenshots (if applicable)
- Related issues or PRs
- Workarounds you've tried
- Any other relevant details

## Possible Solution

<!-- If you have ideas on how to fix this, share them here (optional) -->

## Checklist

- [ ] I have searched for existing issues
- [ ] I have included all relevant information
- [ ] I have included logs and error messages
- [ ] I can consistently reproduce this bug
- [ ] I am using the latest version

#!/bin/bash
# Smoke tests for microservices architecture

GATEWAY_URL="http://localhost:8080"

echo "=========================================="
echo "Microservices Architecture Smoke Tests"
echo "=========================================="
echo ""

# Test 1: Gateway Health
echo "Test 1: Gateway Health Check"
curl -s "$GATEWAY_URL/health" | jq .
echo ""

# Test 2: vLLM Backend (with reduced max_model_len to fit in memory)
echo "Test 2: vLLM Backend - Qwen3-Coder-30B (reduced context)"
curl -X POST "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/qwen3-coder-30b",
    "backend": "vllm",
    "messages": [{"role": "user", "content": "Write a Python hello world"}],
    "max_tokens": 50,
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.9
  }' | jq .
echo ""

# Test 3: Transformers Backend
echo "Test 3: Transformers Backend - Qwen3-Coder-30B"
curl -X POST "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/qwen3-coder-30b",
    "backend": "transformers",
    "messages": [{"role": "user", "content": "Write a Python hello world"}],
    "max_tokens": 50
  }' | jq .
echo ""

# Test 4: llama.cpp Backend (requires GGUF model)
echo "Test 4: llama.cpp Backend - GGUF Model"
curl -X POST "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/qwen3-iq4xs",
    "backend": "llamacpp",
    "messages": [{"role": "user", "content": "Write a Python hello world"}],
    "max_tokens": 50,
    "n_gpu_layers": 40
  }' | jq .
echo ""

# Test 5: Check loaded models
echo "Test 5: Check Loaded Models"
curl -s "$GATEWAY_URL/v1/models/loaded" | jq .
echo ""

echo "=========================================="
echo "Smoke Tests Complete"
echo "=========================================="

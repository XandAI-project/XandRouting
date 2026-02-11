#!/bin/bash
# Test script for Multi-Model LLM Inference API

set -e

API_URL="http://localhost:8080"

echo "==================================="
echo "Multi-Model LLM API Test Suite"
echo "==================================="
echo ""

# Test 1: Health check
echo "Test 1: Health Check"
echo "-----------------------------------"
curl -s "$API_URL/" | python -m json.tool
echo ""
echo ""

# Test 2: List models
echo "Test 2: List Available Models"
echo "-----------------------------------"
curl -s "$API_URL/v1/models" | python -m json.tool
echo ""
echo ""

# Test 3: Invalid model error
echo "Test 3: Invalid Model (Should Return 404)"
echo "-----------------------------------"
curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nonexistent-model",
    "messages": [{"role": "user", "content": "Hello"}]
  }' | python -m json.tool
echo ""
echo ""

# Test 4: GPU Model (CUDA)
echo "Test 4: GPU Model (llama3-8b-cuda)"
echo "-----------------------------------"
curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b-cuda",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }' | python -m json.tool
echo ""
echo ""

# Test 5: GPU+RAM Offload
echo "Test 5: GPU+RAM Offload (mistral-7b-offload)"
echo "-----------------------------------"
curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b-offload",
    "messages": [
      {"role": "user", "content": "Explain AI in 2 sentences."}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }' | python -m json.tool
echo ""
echo ""

# Test 6: CPU-Only
echo "Test 6: CPU-Only Model (qwen2-7b-cpu)"
echo "-----------------------------------"
curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2-7b-cpu",
    "messages": [
      {"role": "user", "content": "Write a haiku about coding."}
    ],
    "max_tokens": 100,
    "temperature": 0.8
  }' | python -m json.tool
echo ""
echo ""

echo "==================================="
echo "Test Suite Complete"
echo "==================================="

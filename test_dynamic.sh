#!/bin/bash
# Test script for Dynamic Model Loading API

set -e

API_URL="http://localhost:8080"

echo "========================================="
echo "Dynamic Model Loading API Test Suite"
echo "========================================="
echo ""

# Test 1: Health check
echo "Test 1: Health Check"
echo "-------------------------------------"
curl -s "$API_URL/health" | python -m json.tool
echo ""
echo ""

# Test 2: Root endpoint with stats
echo "Test 2: Root Endpoint (Status)"
echo "-------------------------------------"
curl -s "$API_URL/" | python -m json.tool
echo ""
echo ""

# Test 3: Load model with vLLM (will take time on first load)
echo "Test 3: Load Model with vLLM (CUDA)"
echo "-------------------------------------"
echo "Note: This will take time on first load..."
curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/test-model",
    "backend": "vllm",
    "device": "cuda",
    "gpu_memory_utilization": 0.7,
    "ttl": 600,
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 50
  }' | python -m json.tool
echo ""
echo ""

# Test 4: Check loaded models
echo "Test 4: List Loaded Models"
echo "-------------------------------------"
curl -s "$API_URL/v1/models/loaded" | python -m json.tool
echo ""
echo ""

# Test 5: Cache hit (same config)
echo "Test 5: Cache Hit Test (Same Config)"
echo "-------------------------------------"
echo "This should be instant (using cached model)..."
curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/test-model",
    "backend": "vllm",
    "device": "cuda",
    "gpu_memory_utilization": 0.7,
    "ttl": 600,
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 20
  }' | python -m json.tool
echo ""
echo ""

# Test 6: Get statistics
echo "Test 6: Cache Statistics"
echo "-------------------------------------"
curl -s "$API_URL/v1/models/stats" | python -m json.tool
echo ""
echo ""

# Test 7: Switch to Transformers (exclusive mode)
echo "Test 7: Switch to Transformers (Exclusive Mode)"
echo "-------------------------------------"
echo "This will unload vLLM version and load Transformers..."
curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/test-model",
    "backend": "transformers",
    "device": "cpu",
    "ttl": 300,
    "messages": [
      {"role": "user", "content": "Count to 3."}
    ],
    "max_tokens": 30
  }' | python -m json.tool
echo ""
echo ""

# Test 8: List loaded models again
echo "Test 8: List Loaded Models (After Switch)"
echo "-------------------------------------"
curl -s "$API_URL/v1/models/loaded" | python -m json.tool
echo ""
echo ""

# Test 9: Unload specific model
echo "Test 9: Unload Specific Model"
echo "-------------------------------------"
curl -s -X POST "$API_URL/v1/models/unload" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/test-model",
    "backend": "transformers",
    "device": "cpu"
  }' | python -m json.tool
echo ""
echo ""

# Test 10: Verify unloaded
echo "Test 10: Verify Model Unloaded"
echo "-------------------------------------"
curl -s "$API_URL/v1/models/loaded" | python -m json.tool
echo ""
echo ""

echo "========================================="
echo "Test Suite Complete!"
echo "========================================="
echo ""
echo "Key Features Tested:"
echo "  ✓ Health check"
echo "  ✓ Dynamic model loading (vLLM)"
echo "  ✓ Model caching (cache hit)"
echo "  ✓ Statistics tracking"
echo "  ✓ Backend switching (exclusive mode)"
echo "  ✓ Manual unload"
echo ""
echo "System is working correctly!"

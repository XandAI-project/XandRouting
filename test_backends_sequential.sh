#!/bin/bash
# Sequential Backend Testing Script
# Tests each backend (vLLM, Transformers, llama.cpp) with automatic unload between tests

set -e

GATEWAY_URL="http://localhost:8080"

echo "=========================================="
echo "Sequential Backend Testing"
echo "Testing each backend with automatic unload"
echo "=========================================="

# Function to test backend
test_backend() {
    BACKEND=$1
    MODEL=$2
    EXTRA_PARAMS=$3
    
    echo ""
    echo "----------------------------------------"
    echo "Testing: $BACKEND"
    echo "----------------------------------------"
    
    # Test generation
    START_TIME=$(date +%s)
    RESPONSE=$(curl -s -X POST "$GATEWAY_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"backend\": \"$BACKEND\",
            \"device\": \"cuda\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}],
            \"max_tokens\": 20
            $EXTRA_PARAMS
        }")
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # Check if successful
    if echo "$RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
        echo "✓ $BACKEND: SUCCESS (${DURATION}s)"
        echo "Response:"
        echo "$RESPONSE" | jq '.choices[0].message.content'
    else
        echo "✗ $BACKEND: FAILED"
        echo "Error response:"
        echo "$RESPONSE" | jq '.'
        return 1
    fi
    
    # Unload model
    echo ""
    echo "Unloading $BACKEND..."
    UNLOAD_RESPONSE=$(curl -s -X POST "$GATEWAY_URL/v1/models/unload" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL\", \"backend\": \"$BACKEND\"}")
    
    echo "$UNLOAD_RESPONSE" | jq '.'
    
    # Wait for cleanup
    echo "Waiting for cleanup..."
    sleep 2
}

# Check if gateway is accessible
echo "Checking gateway health..."
if ! curl -s "$GATEWAY_URL/health" > /dev/null; then
    echo "Error: Gateway not accessible at $GATEWAY_URL"
    exit 1
fi
echo "Gateway is healthy"

# Test each backend sequentially
echo ""
echo "Starting sequential backend tests..."

# Test 1: vLLM Backend
test_backend "vllm" "/models/qwen3-coder-30b" ", \"max_model_len\": 8192, \"gpu_memory_utilization\": 0.4"

# Test 2: Transformers Backend
test_backend "transformers" "/models/qwen3-coder-30b" ""

# Test 3: llama.cpp Backend
test_backend "llamacpp" "/models/qwen3-iq4xs/Qwen3-Coder-30B-A3B-Instruct-IQ4_XS.gguf" ", \"n_ctx\": 4096, \"n_gpu_layers\": 20"

echo ""
echo "=========================================="
echo "All Backend Tests Complete"
echo "=========================================="
echo ""

# Show final status
echo "Final status check:"
curl -s "$GATEWAY_URL/v1/models/loaded" | jq '.'

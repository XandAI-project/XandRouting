# Smoke tests for microservices architecture

$GATEWAY_URL = "http://localhost:8080"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Microservices Architecture Smoke Tests" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Gateway Health
Write-Host "Test 1: Gateway Health Check" -ForegroundColor Yellow
Invoke-RestMethod -Uri "$GATEWAY_URL/health" -Method Get | ConvertTo-Json -Depth 10
Write-Host ""

# Test 2: vLLM Backend (with reduced max_model_len to fit in memory)
Write-Host "Test 2: vLLM Backend - Qwen3-Coder-30B (reduced context)" -ForegroundColor Yellow
$vllmBody = @{
    model = "/models/qwen3-coder-30b"
    backend = "vllm"
    device = "cuda"
    messages = @(
        @{
            role = "user"
            content = "Write a Python hello world"
        }
    )
    max_tokens = 50
    max_model_len = 8192
    gpu_memory_utilization = 0.9
} | ConvertTo-Json -Depth 10

try {
    Invoke-RestMethod -Uri "$GATEWAY_URL/v1/chat/completions" -Method Post -Body $vllmBody -ContentType "application/json" | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 3: Transformers Backend
Write-Host "Test 3: Transformers Backend - Qwen3-Coder-30B" -ForegroundColor Yellow
$transformersBody = @{
    model = "/models/qwen3-coder-30b"
    backend = "transformers"
    device = "cuda"
    messages = @(
        @{
            role = "user"
            content = "Write a Python hello world"
        }
    )
    max_tokens = 50
} | ConvertTo-Json -Depth 10

try {
    Invoke-RestMethod -Uri "$GATEWAY_URL/v1/chat/completions" -Method Post -Body $transformersBody -ContentType "application/json" | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 4: llama.cpp Backend (requires GGUF model)
Write-Host "Test 4: llama.cpp Backend - GGUF Model" -ForegroundColor Yellow
$llamacppBody = @{
    model = "/models/qwen3-iq4xs"
    backend = "llamacpp"
    device = "cuda"
    messages = @(
        @{
            role = "user"
            content = "Write a Python hello world"
        }
    )
    max_tokens = 50
    n_gpu_layers = 40
} | ConvertTo-Json -Depth 10

try {
    Invoke-RestMethod -Uri "$GATEWAY_URL/v1/chat/completions" -Method Post -Body $llamacppBody -ContentType "application/json" | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 5: Check loaded models
Write-Host "Test 5: Check Loaded Models" -ForegroundColor Yellow
Invoke-RestMethod -Uri "$GATEWAY_URL/v1/models/loaded" -Method Get | ConvertTo-Json -Depth 10
Write-Host ""

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Smoke Tests Complete" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

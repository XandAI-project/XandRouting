# PowerShell Test Script for Multi-Model LLM Inference API

$API_URL = "http://localhost:8080"

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Multi-Model LLM API Test Suite" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Health check
Write-Host "Test 1: Health Check" -ForegroundColor Yellow
Write-Host "-----------------------------------"
$response = Invoke-RestMethod -Uri "$API_URL/" -Method Get
$response | ConvertTo-Json -Depth 10
Write-Host ""

# Test 2: List models
Write-Host "Test 2: List Available Models" -ForegroundColor Yellow
Write-Host "-----------------------------------"
$response = Invoke-RestMethod -Uri "$API_URL/v1/models" -Method Get
$response | ConvertTo-Json -Depth 10
Write-Host ""

# Test 3: Invalid model error
Write-Host "Test 3: Invalid Model (Should Return 404)" -ForegroundColor Yellow
Write-Host "-----------------------------------"
try {
    $body = @{
        model = "nonexistent-model"
        messages = @(
            @{role = "user"; content = "Hello"}
        )
    } | ConvertTo-Json -Depth 10
    
    $response = Invoke-RestMethod -Uri "$API_URL/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error (Expected): $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 4: GPU Model (CUDA)
Write-Host "Test 4: GPU Model (llama3-8b-cuda)" -ForegroundColor Yellow
Write-Host "-----------------------------------"
$body = @{
    model = "llama3-8b-cuda"
    messages = @(
        @{role = "system"; content = "You are a helpful assistant."}
        @{role = "user"; content = "Say hello in one sentence."}
    )
    max_tokens = 50
    temperature = 0.7
} | ConvertTo-Json -Depth 10

try {
    $response = Invoke-RestMethod -Uri "$API_URL/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 5: GPU+RAM Offload
Write-Host "Test 5: GPU+RAM Offload (mistral-7b-offload)" -ForegroundColor Yellow
Write-Host "-----------------------------------"
$body = @{
    model = "mistral-7b-offload"
    messages = @(
        @{role = "user"; content = "Explain AI in 2 sentences."}
    )
    max_tokens = 100
    temperature = 0.7
} | ConvertTo-Json -Depth 10

try {
    $response = Invoke-RestMethod -Uri "$API_URL/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 6: CPU-Only
Write-Host "Test 6: CPU-Only Model (qwen2-7b-cpu)" -ForegroundColor Yellow
Write-Host "-----------------------------------"
$body = @{
    model = "qwen2-7b-cpu"
    messages = @(
        @{role = "user"; content = "Write a haiku about coding."}
    )
    max_tokens = 100
    temperature = 0.8
} | ConvertTo-Json -Depth 10

try {
    $response = Invoke-RestMethod -Uri "$API_URL/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Test Suite Complete" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan

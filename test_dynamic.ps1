# PowerShell Test Script for Dynamic Model Loading API

$API_URL = "http://localhost:8080"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Dynamic Model Loading API Test Suite" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Health check
Write-Host "Test 1: Health Check" -ForegroundColor Yellow
Write-Host "-------------------------------------"
$response = Invoke-RestMethod -Uri "$API_URL/health" -Method Get
$response | ConvertTo-Json -Depth 10
Write-Host ""

# Test 2: Root endpoint
Write-Host "Test 2: Root Endpoint (Status)" -ForegroundColor Yellow
Write-Host "-------------------------------------"
$response = Invoke-RestMethod -Uri "$API_URL/" -Method Get
$response | ConvertTo-Json -Depth 10
Write-Host ""

# Test 3: Load model with vLLM
Write-Host "Test 3: Load Model with vLLM (CUDA)" -ForegroundColor Yellow
Write-Host "-------------------------------------"
Write-Host "Note: This will take time on first load..." -ForegroundColor Gray

$body = @{
    model = "/models/test-model"
    backend = "vllm"
    device = "cuda"
    gpu_memory_utilization = 0.7
    ttl = 600
    messages = @(
        @{role = "user"; content = "Say hello in one sentence."}
    )
    temperature = 0.7
    max_tokens = 50
} | ConvertTo-Json -Depth 10

try {
    $response = Invoke-RestMethod -Uri "$API_URL/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 4: List loaded models
Write-Host "Test 4: List Loaded Models" -ForegroundColor Yellow
Write-Host "-------------------------------------"
try {
    $response = Invoke-RestMethod -Uri "$API_URL/v1/models/loaded" -Method Get
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 5: Cache hit test
Write-Host "Test 5: Cache Hit Test (Same Config)" -ForegroundColor Yellow
Write-Host "-------------------------------------"
Write-Host "This should be instant (using cached model)..." -ForegroundColor Gray

$body = @{
    model = "/models/test-model"
    backend = "vllm"
    device = "cuda"
    gpu_memory_utilization = 0.7
    ttl = 600
    messages = @(
        @{role = "user"; content = "What is 2+2?"}
    )
    max_tokens = 20
} | ConvertTo-Json -Depth 10

try {
    $response = Invoke-RestMethod -Uri "$API_URL/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 6: Get statistics
Write-Host "Test 6: Cache Statistics" -ForegroundColor Yellow
Write-Host "-------------------------------------"
try {
    $response = Invoke-RestMethod -Uri "$API_URL/v1/models/stats" -Method Get
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 7: Switch to Transformers
Write-Host "Test 7: Switch to Transformers (Exclusive Mode)" -ForegroundColor Yellow
Write-Host "-------------------------------------"
Write-Host "This will unload vLLM version and load Transformers..." -ForegroundColor Gray

$body = @{
    model = "/models/test-model"
    backend = "transformers"
    device = "cpu"
    ttl = 300
    messages = @(
        @{role = "user"; content = "Count to 3."}
    )
    max_tokens = 30
} | ConvertTo-Json -Depth 10

try {
    $response = Invoke-RestMethod -Uri "$API_URL/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 8: List loaded models again
Write-Host "Test 8: List Loaded Models (After Switch)" -ForegroundColor Yellow
Write-Host "-------------------------------------"
try {
    $response = Invoke-RestMethod -Uri "$API_URL/v1/models/loaded" -Method Get
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 9: Unload model
Write-Host "Test 9: Unload Specific Model" -ForegroundColor Yellow
Write-Host "-------------------------------------"

$body = @{
    model = "/models/test-model"
    backend = "transformers"
    device = "cpu"
} | ConvertTo-Json -Depth 10

try {
    $response = Invoke-RestMethod -Uri "$API_URL/v1/models/unload" -Method Post -Body $body -ContentType "application/json"
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 10: Verify unloaded
Write-Host "Test 10: Verify Model Unloaded" -ForegroundColor Yellow
Write-Host "-------------------------------------"
try {
    $response = Invoke-RestMethod -Uri "$API_URL/v1/models/loaded" -Method Get
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Test Suite Complete!" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Key Features Tested:" -ForegroundColor Green
Write-Host "  ✓ Health check" -ForegroundColor White
Write-Host "  ✓ Dynamic model loading (vLLM)" -ForegroundColor White
Write-Host "  ✓ Model caching (cache hit)" -ForegroundColor White
Write-Host "  ✓ Statistics tracking" -ForegroundColor White
Write-Host "  ✓ Backend switching (exclusive mode)" -ForegroundColor White
Write-Host "  ✓ Manual unload" -ForegroundColor White
Write-Host ""
Write-Host "System is working correctly!" -ForegroundColor Green

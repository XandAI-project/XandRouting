from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import os
from typing import Dict, Any
import httpx

from models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ModelConfig,
    ErrorResponse,
    LoadedModelsResponse,
    ModelStatsResponse,
    UnloadModelRequest,
    ModelInventoryResponse
)
from download_models import DownloadRequest, DownloadJob, DownloadListResponse
from download_manager import download_manager
from model_scanner import scan_models_directory
from engine_router import engine_router
from cors_config import add_cors_middleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Dynamic Multi-Model LLM Gateway",
    description="Gateway service that routes requests to backend engine workers (vLLM, Transformers, llama.cpp)",
    version="2.0.0-microservices"
)

# Add CORS middleware
add_cors_middleware(app)


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("=" * 50)
    logger.info("Dynamic Multi-Model LLM Gateway Starting")
    logger.info("=" * 50)
    logger.info("Architecture: Microservices")
    logger.info("Features:")
    logger.info("  - HTTP Gateway (request routing)")
    logger.info("  - vLLM worker service")
    logger.info("  - Transformers worker service")
    logger.info("  - llama.cpp worker service")
    logger.info("  - Model download management")
    logger.info("=" * 50)
    
    # Check health of all engines
    health_status = await engine_router.health_check_all()
    logger.info("Engine health status:")
    for backend, status in health_status.items():
        logger.info(f"  {backend}: {status['status']}")


@app.get("/")
async def root():
    """Health check and status endpoint"""
    # Check health of all engines
    engine_health = await engine_router.health_check_all()
    
    return {
        "status": "online",
        "service": "Dynamic Multi-Model LLM Gateway",
        "version": "2.0.0-microservices",
        "architecture": "microservices",
        "features": [
            "HTTP Gateway (request routing)",
            "vLLM worker service",
            "Transformers worker service",
            "llama.cpp worker service",
            "Model download management",
            "CORS support"
        ],
        "engines": engine_health
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    # Check health of all engines
    engine_health = await engine_router.health_check_all()
    
    all_healthy = all(status["status"] == "healthy" for status in engine_health.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "gateway": "healthy",
        "engines": engine_health
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint - proxies to backend workers
    
    Supports both streaming and non-streaming responses.
    
    Request body includes model configuration:
    - model: Path to model directory (e.g., /models/gpt-oss)
    - backend: "vllm", "transformers", or "llamacpp"
    - device: "cuda" or "cpu"
    - gpu_memory_utilization: 0.1-1.0 (optional, default 0.7)
    - ttl: Time-to-live in seconds (optional, default 300)
    - stream: Enable streaming response (optional, default False)
    - Plus all standard OpenAI parameters
    """
    try:
        backend = request.backend
        
        logger.info(f"Routing request to {backend.value} engine: {request.model} (stream={request.stream})")
        
        # Prepare request body for worker
        request_body = request.model_dump()
        
        # Handle streaming vs non-streaming
        if request.stream:
            # Streaming response - proxy stream from worker
            async def proxy_stream():
                try:
                    async for chunk in engine_router.proxy_streaming_request(
                        backend=backend,
                        endpoint="/v1/chat/completions",
                        json_data=request_body,
                        timeout=300.0
                    ):
                        yield chunk
                except httpx.HTTPError as e:
                    logger.error(f"Error proxying streaming request: {e}")
                    # Send error as SSE
                    error_data = {
                        "error": {
                            "message": f"Backend service error: {str(e)}",
                            "type": "backend_error"
                        }
                    }
                    yield f"data: {error_data}\n\n"
            
            return StreamingResponse(
                proxy_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        else:
            # Non-streaming response - proxy to worker
            response = await engine_router.proxy_request(
                backend=backend,
                endpoint="/v1/chat/completions",
                method="POST",
                json_data=request_body,
                timeout=300.0
            )
            
            # Check response status
            if response.status_code != 200:
                logger.error(f"Worker returned error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Backend service error: {response.text}"
                )
            
            logger.info(f"Generation complete for {request.model}")
            
            return response.json()
        
    except httpx.HTTPError as e:
        logger.error(f"HTTP error communicating with backend: {e}")
        raise HTTPException(status_code=502, detail=f"Backend service error: {str(e)}")
    
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/v1/models/loaded", response_model=LoadedModelsResponse)
async def list_loaded_models():
    """
    List currently loaded models across all engine workers
    Aggregates loaded models from all backend services
    """
    all_loaded_models = []
    
    # Query each engine for loaded models
    for backend in [request.backend.VLLM, request.backend.TRANSFORMERS, request.backend.LLAMACPP]:
        try:
            response = await engine_router.proxy_request(
                backend=backend,
                endpoint="/health",
                method="GET",
                timeout=5.0
            )
            
            if response.status_code == 200:
                health_data = response.json()
                models = health_data.get("models", [])
                
                # Convert to CachedModel format (simplified)
                for model_path in models:
                    from models import CachedModel
                    from datetime import datetime
                    all_loaded_models.append(CachedModel(
                        cache_key=f"{model_path}:{backend.value}",
                        model_path=model_path,
                        backend=backend,
                        device="cuda",
                        gpu_memory_utilization=0.7,
                        loaded_at=datetime.now(),
                        last_used=datetime.now(),
                        expires_at=None,
                        ttl=0
                    ))
        except Exception as e:
            logger.error(f"Error querying {backend.value} for loaded models: {e}")
    
    return LoadedModelsResponse(
        loaded_models=all_loaded_models,
        total_count=len(all_loaded_models)
    )


@app.post("/v1/models/unload")
async def unload_model(request: UnloadModelRequest):
    """
    Manually unload a specific model from backend workers
    
    Request body:
    - model: Model path to unload
    - backend: Optional backend filter
    - device: Optional device filter
    """
    try:
        unload_request = {
            "model_path": request.model
        }
        
        if request.backend:
            # Unload from specific backend
            response = await engine_router.proxy_request(
                backend=request.backend,
                endpoint="/v1/models/unload",
                method="POST",
                json_data=unload_request,
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
        else:
            # Unload from all backends
            results = []
            for backend in [request.backend.VLLM, request.backend.TRANSFORMERS, request.backend.LLAMACPP]:
                try:
                    response = await engine_router.proxy_request(
                        backend=backend,
                        endpoint="/v1/models/unload",
                        method="POST",
                        json_data=unload_request,
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        results.append({"backend": backend.value, "status": "unloaded"})
                except Exception as e:
                    logger.warning(f"Failed to unload from {backend.value}: {e}")
            
            return {
                "status": "unloaded",
                "model_path": request.model,
                "results": results,
                "count": len(results)
            }
    
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/models/unload-all")
async def unload_all_models():
    """
    Unload all currently loaded models from all backend workers
    """
    total_count = 0
    results = []
    
    # Unload from all backends
    from models import BackendType
    for backend in [BackendType.VLLM, BackendType.TRANSFORMERS, BackendType.LLAMACPP]:
        try:
            # Note: Workers don't have an unload-all endpoint, so we skip this
            # Could be implemented by querying health for models and unloading each
            logger.info(f"Skipping unload-all for {backend.value} (not implemented in workers)")
        except Exception as e:
            logger.warning(f"Failed to unload all from {backend.value}: {e}")
    
    return {
        "status": "unload_all_not_implemented",
        "message": "Individual model unloading is supported via /v1/models/unload endpoint",
        "count": total_count
    }


@app.get("/v1/models/stats", response_model=ModelStatsResponse)
async def model_stats():
    """
    Get aggregated statistics from all backend workers
    """
    # Aggregate stats from all engines
    total_models = 0
    
    from models import BackendType
    for backend in [BackendType.VLLM, BackendType.TRANSFORMERS, BackendType.LLAMACPP]:
        try:
            response = await engine_router.proxy_request(
                backend=backend,
                endpoint="/health",
                method="GET",
                timeout=5.0
            )
            
            if response.status_code == 200:
                health_data = response.json()
                total_models += health_data.get("loaded_models", 0)
        except Exception as e:
            logger.error(f"Error getting stats from {backend.value}: {e}")
    
    return ModelStatsResponse(
        total_models_loaded=total_models,
        active_models=total_models,
        total_requests_served=0,  # Not tracked in microservices mode
        cache_hits=0,
        cache_misses=0,
        hit_rate=0.0,
        estimated_gpu_memory_mb=None,
        estimated_cpu_memory_mb=None
    )


@app.get("/v1/models/inventory", response_model=ModelInventoryResponse)
async def list_model_inventory():
    """
    List all downloaded models in /models/ directory
    
    Returns comprehensive information about each model:
    - Model name and path
    - File details (sizes, types, quantization)
    - Total size in GB
    - Recommended backend(s)
    - Config metadata if available
    """
    try:
        inventory = await scan_models_directory("/models")
        
        return ModelInventoryResponse(
            models=inventory["models"],
            total_models=len(inventory["models"]),
            total_size_gb=inventory["total_size_gb"]
        )
    
    except Exception as e:
        logger.error(f"Error scanning model inventory: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to scan model inventory: {str(e)}"
        )


# ============================================================================
# Model Download Endpoints
# ============================================================================

@app.post("/v1/models/download", response_model=DownloadJob)
async def start_download(request: DownloadRequest):
    """
    Start background download of HuggingFace model
    
    Request body:
    - url: Full HuggingFace URL (e.g., https://huggingface.co/org/model)
    - destination: Optional target directory name in /models/
    - include: Optional file patterns to include (e.g., ["*.gguf"])
    - exclude: Optional file patterns to exclude
    - quantization: Optional GGUF quantization type (e.g., "IQ4_XS")
    
    Returns job_id for status polling
    """
    try:
        job_id = download_manager.create_job(request)
        job = download_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=500, detail="Failed to create download job")
        
        logger.info(f"Download job created: {job_id}")
        
        return job
        
    except ValueError as e:
        logger.error(f"Invalid download request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error creating download job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create download job: {str(e)}")


@app.get("/v1/models/download/{job_id}", response_model=DownloadJob)
async def get_download_status(job_id: str):
    """
    Get download job status
    
    Poll this endpoint to track download progress
    
    Returns:
    - status: pending, downloading, completed, failed, cancelled
    - Additional fields depending on status
    """
    job = download_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Download job not found: {job_id}")
    
    return job


@app.get("/v1/models/download", response_model=DownloadListResponse)
async def list_downloads():
    """
    List all download jobs
    
    Returns list of all download jobs with their current status
    """
    jobs = download_manager.list_jobs()
    
    return DownloadListResponse(
        downloads=jobs,
        total=len(jobs)
    )


@app.delete("/v1/models/download/{job_id}")
async def cancel_download(job_id: str):
    """
    Cancel a running download job
    
    Can only cancel jobs that are pending or downloading
    """
    success = download_manager.cancel_job(job_id)
    
    if not success:
        job = download_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Download job not found: {job_id}")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job with status: {job.status}"
            )
    
    return {
        "status": "cancelled",
        "job_id": job_id,
        "message": "Download job cancelled successfully"
    }


@app.post("/v1/models/verify-repo")
async def verify_repository(request: DownloadRequest):
    """
    Verify if a HuggingFace repository exists and is accessible
    
    Use this to test connectivity and repository URL before downloading
    """
    try:
        from huggingface_hub import list_repo_files, repo_info
        
        # Parse repo ID
        repo_id = download_manager._parse_repo_id(request.url)
        
        logger.info(f"Verifying repository: {repo_id}")
        
        # Try to get repo info
        info = repo_info(repo_id, token=None)
        
        # Try to list files
        files = list_repo_files(repo_id, token=None)
        
        # Filter by patterns if provided
        matched_files = files
        if request.include:
            import fnmatch
            matched_files = [f for f in files if any(fnmatch.fnmatch(f, pattern) for pattern in request.include)]
        
        if request.quantization:
            matched_files = [f for f in files if request.quantization in f and f.endswith('.gguf')]
        
        return {
            "status": "accessible",
            "repo_id": repo_id,
            "repo_type": info.repo_type if hasattr(info, 'repo_type') else "model",
            "total_files": len(files),
            "matched_files": len(matched_files),
            "sample_files": matched_files[:10] if matched_files else files[:10],
            "message": f"Repository is accessible with {len(matched_files)} matching files"
        }
        
    except Exception as e:
        logger.error(f"Repository verification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Repository verification failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "details": str(exc)
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

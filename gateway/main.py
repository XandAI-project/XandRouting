from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import os
from typing import Dict, Any

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
from model_cache import model_cache
from inference import inference_engine
from download_models import DownloadRequest, DownloadJob, DownloadListResponse
from download_manager import download_manager
from model_scanner import scan_models_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Dynamic Multi-Model LLM Inference API",
    description="Load and run any LLM model dynamically with vLLM or Transformers",
    version="2.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("=" * 50)
    logger.info("Dynamic Multi-Model LLM API Starting")
    logger.info("=" * 50)
    logger.info("Features:")
    logger.info("  - Dynamic model loading (vLLM + Transformers)")
    logger.info("  - Automatic TTL-based unloading")
    logger.info("  - Exclusive mode (same model, different configs)")
    logger.info("  - OpenAI-compatible API")
    logger.info("=" * 50)


@app.get("/")
async def root():
    """Health check and status endpoint"""
    stats = model_cache.get_stats()
    
    return {
        "status": "online",
        "service": "Dynamic Multi-Model LLM Gateway",
        "version": "2.0.0",
        "features": [
            "Dynamic model loading",
            "vLLM backend support",
            "Transformers backend support",
            "TTL-based auto-unload",
            "Exclusive mode"
        ],
        "loaded_models": model_cache.get_model_count(),
        "cache_stats": stats
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with dynamic model loading
    
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
        # Create model configuration
        config = ModelConfig(
            model=request.model,
            backend=request.backend,
            device=request.device,
            gpu_memory_utilization=request.gpu_memory_utilization,
            ttl=request.ttl,
            n_gpu_layers=request.n_gpu_layers if request.backend == "llamacpp" else -1,
            n_ctx=request.n_ctx if request.backend == "llamacpp" else 2048,
            max_model_len=request.max_model_len
        )
        
        logger.info(f"Request for model: {config.model} ({config.backend.value} on {config.device.value}, stream={request.stream})")
        
        # Get or load model (from cache)
        model_wrapper = await model_cache.get_or_load(config)
        
        # Prepare generation parameters
        gen_params = {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stop": request.stop,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty
        }
        
        # Handle streaming vs non-streaming
        if request.stream:
            # Streaming response
            async def generate_sse():
                async for chunk in inference_engine.generate_stream(
                    model_wrapper=model_wrapper,
                    messages=request.messages,
                    params=gen_params
                ):
                    # Format as SSE
                    chunk_json = chunk.model_dump_json()
                    yield f"data: {chunk_json}\n\n"
                
                # Send [DONE] message
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        else:
            # Non-streaming response
            response = await inference_engine.generate(
                model_wrapper=model_wrapper,
                messages=request.messages,
                params=gen_params
            )
            
            logger.info(f"Generation complete for {config.model}")
            
            return response
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except RuntimeError as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/v1/models/loaded", response_model=LoadedModelsResponse)
async def list_loaded_models():
    """
    List currently loaded models with TTL information
    """
    loaded = model_cache.get_loaded_models()
    
    return LoadedModelsResponse(
        loaded_models=loaded,
        total_count=len(loaded)
    )


@app.post("/v1/models/unload")
async def unload_model(request: UnloadModelRequest):
    """
    Manually unload a specific model or all models with a given path
    
    Request body:
    - model: Model path to unload
    - backend: Optional backend filter
    - device: Optional device filter
    """
    if request.backend and request.device:
        # Unload specific configuration
        config = ModelConfig(
            model=request.model,
            backend=request.backend,
            device=request.device
        )
        cache_key = config.generate_cache_key()
        
        success = model_cache.unload_model(cache_key)
        
        if success:
            return {"status": "unloaded", "cache_key": cache_key}
        else:
            raise HTTPException(status_code=404, detail=f"Model not found: {cache_key}")
    
    else:
        # Unload all configurations of this model
        count = model_cache.unload_by_path(request.model)
        
        return {
            "status": "unloaded",
            "model_path": request.model,
            "count": count
        }


@app.post("/v1/models/unload-all")
async def unload_all_models():
    """
    Unload all currently loaded models
    """
    count = model_cache.unload_all()
    
    return {
        "status": "unloaded_all",
        "count": count
    }


@app.get("/v1/models/stats", response_model=ModelStatsResponse)
async def model_stats():
    """
    Get cache statistics and memory usage information
    """
    stats = model_cache.get_stats()
    
    return ModelStatsResponse(**stats)


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

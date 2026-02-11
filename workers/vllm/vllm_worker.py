from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import logging
import os
import time
import asyncio
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM Engine Worker")

# Global model cache
loaded_models: Dict[str, Any] = {}
model_metadata: Dict[str, Dict[str, Any]] = {}


class LoadModelRequest(BaseModel):
    """Request to load a model"""
    model_path: str = Field(..., description="Path to model directory")
    gpu_memory_utilization: float = Field(default=0.7, ge=0.1, le=1.0)
    max_model_len: Optional[int] = Field(default=None, description="Maximum context length")
    ttl: int = Field(default=300, ge=0, description="Time-to-live in seconds")


class UnloadModelRequest(BaseModel):
    """Request to unload a model"""
    model_path: str = Field(..., description="Path to model directory to unload")


class ChatMessage(BaseModel):
    """Chat message"""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(..., description="Model path")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=512, ge=1)
    stream: Optional[bool] = Field(default=False)
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    # vLLM-specific parameters (optional)
    gpu_memory_utilization: Optional[float] = Field(default=0.7, ge=0.1, le=1.0)
    max_model_len: Optional[int] = None
    ttl: Optional[int] = Field(default=300, ge=0)


def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert OpenAI message format to a prompt string"""
    prompt_parts = []
    
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    
    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)


async def load_model_if_needed(
    model_path: str,
    gpu_memory_utilization: float = 0.7,
    max_model_len: Optional[int] = None,
    ttl: int = 300
) -> Any:
    """Load model into vLLM if not already loaded"""
    if model_path in loaded_models:
        logger.info(f"Model already loaded: {model_path}")
        # Update last_used time
        model_metadata[model_path]["last_used"] = datetime.now()
        return loaded_models[model_path]
    
    logger.info(f"Loading model with vLLM: {model_path}")
    logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
    if max_model_len:
        logger.info(f"Max model length: {max_model_len}")
    
    try:
        from vllm import LLM
        
        # Load model with vLLM
        llm_kwargs = {
            "model": model_path,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
        }
        
        if max_model_len:
            llm_kwargs["max_model_len"] = max_model_len
        
        llm = LLM(**llm_kwargs)
        
        # Store in cache
        loaded_models[model_path] = llm
        model_metadata[model_path] = {
            "loaded_at": datetime.now(),
            "last_used": datetime.now(),
            "ttl": ttl,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len
        }
        
        logger.info(f"Model loaded successfully: {model_path}")
        return llm
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


async def unload_model(model_path: str) -> bool:
    """Unload a model from memory"""
    if model_path not in loaded_models:
        return False
    
    logger.info(f"Unloading model: {model_path}")
    
    try:
        # Remove from cache
        del loaded_models[model_path]
        del model_metadata[model_path]
        
        # Force garbage collection
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"Model unloaded: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error unloading model: {e}", exc_info=True)
        return False


async def cleanup_expired_models():
    """Background task to cleanup expired models based on TTL"""
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            now = datetime.now()
            models_to_unload = []
            
            for model_path, metadata in model_metadata.items():
                ttl = metadata["ttl"]
                if ttl == 0:
                    continue  # No expiry
                
                last_used = metadata["last_used"]
                expiry_time = last_used + timedelta(seconds=ttl)
                
                if now > expiry_time:
                    logger.info(f"Model expired: {model_path}")
                    models_to_unload.append(model_path)
            
            # Unload expired models
            for model_path in models_to_unload:
                await unload_model(model_path)
                
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("=" * 50)
    logger.info("vLLM Engine Worker Starting")
    logger.info("=" * 50)
    
    # Start cleanup task
    asyncio.create_task(cleanup_expired_models())


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine": "vllm",
        "loaded_models": len(loaded_models),
        "models": list(loaded_models.keys())
    }


@app.post("/v1/models/load")
async def load_model_endpoint(request: LoadModelRequest):
    """Load a model into memory"""
    try:
        await load_model_if_needed(
            model_path=request.model_path,
            gpu_memory_utilization=request.gpu_memory_utilization,
            max_model_len=request.max_model_len,
            ttl=request.ttl
        )
        
        return {
            "status": "loaded",
            "model_path": request.model_path,
            "gpu_memory_utilization": request.gpu_memory_utilization
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/models/unload")
async def unload_model_endpoint(request: UnloadModelRequest):
    """Unload a model from memory"""
    success = await unload_model(request.model_path)
    
    if success:
        return {
            "status": "unloaded",
            "model_path": request.model_path
        }
    else:
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint
    Loads model on-demand if not already loaded
    """
    try:
        # Load model if needed
        llm = await load_model_if_needed(
            model_path=request.model,
            gpu_memory_utilization=request.gpu_memory_utilization or 0.7,
            max_model_len=request.max_model_len,
            ttl=request.ttl or 300
        )
        
        # Convert messages to prompt
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        prompt = messages_to_prompt(messages_dict)
        
        logger.info(f"Generating with vLLM: max_tokens={request.max_tokens}, temperature={request.temperature}")
        
        # Import vLLM SamplingParams
        from vllm import SamplingParams
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty
        )
        
        if request.stream:
            # Streaming response
            async def generate_stream():
                chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
                
                # First chunk with role
                yield f'data: {{"id": "{chunk_id}", "object": "chat.completion.chunk", "created": {int(time.time())}, "model": "{request.model}", "choices": [{{"index": 0, "delta": {{"role": "assistant"}}, "finish_reason": null}}]}}\n\n'
                
                # Generate (vLLM doesn't support true streaming in this context, so we simulate it)
                outputs = llm.generate([prompt], sampling_params)
                
                for output in outputs:
                    text = output.outputs[0].text
                    
                    # Send text chunk
                    yield f'data: {{"id": "{chunk_id}", "object": "chat.completion.chunk", "created": {int(time.time())}, "model": "{request.model}", "choices": [{{"index": 0, "delta": {{"content": {repr(text)}}}, "finish_reason": null}}]}}\n\n'
                
                # Final chunk
                yield f'data: {{"id": "{chunk_id}", "object": "chat.completion.chunk", "created": {int(time.time())}, "model": "{request.model}", "choices": [{{"index": 0, "delta": {{}}, "finish_reason": "stop"}}]}}\n\n'
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        else:
            # Non-streaming response
            outputs = llm.generate([prompt], sampling_params)
            
            # Extract output
            output = outputs[0]
            generated_text = output.outputs[0].text
            
            # Estimate token counts
            prompt_tokens = len(prompt.split())
            completion_tokens = len(generated_text.split())
            
            return {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": output.outputs[0].finish_reason or "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting vLLM worker on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

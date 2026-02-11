from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch
import logging
import os
import time
import asyncio
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Transformers Engine Worker")

# Global model cache
loaded_models: Dict[str, Dict[str, Any]] = {}
model_metadata: Dict[str, Dict[str, Any]] = {}


class LoadModelRequest(BaseModel):
    """Request to load a model"""
    model_path: str = Field(..., description="Path to model directory")
    device: Literal["cuda", "cpu"] = Field(default="cuda", description="Device to load on")
    torch_dtype: Optional[str] = Field(default="auto", description="Torch dtype (auto, float16, float32)")
    ttl: int = Field(default=300, ge=0, description="Time-to-live in seconds")


class UnloadModelRequest(BaseModel):
    """Request to unload a model"""
    model_path: str = Field(..., description="Path to model to unload")


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
    # Transformers-specific parameters
    device: Optional[Literal["cuda", "cpu"]] = Field(default="cuda")
    torch_dtype: Optional[str] = Field(default="auto")
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
    device: str = "cuda",
    torch_dtype: str = "auto",
    ttl: int = 300
) -> Dict[str, Any]:
    """Load model with Transformers if not already loaded"""
    if model_path in loaded_models:
        logger.info(f"Model already loaded: {model_path}")
        # Update last_used time
        model_metadata[model_path]["last_used"] = datetime.now()
        return loaded_models[model_path]
    
    logger.info(f"Loading model with Transformers: {model_path}")
    logger.info(f"Device: {device}, Dtype: {torch_dtype}")
    
    try:
        # Determine torch dtype
        if torch_dtype == "auto":
            dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.float32
        
        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
            dtype = torch.float32
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
        
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = dtype
        else:
            model_kwargs["device_map"] = "cpu"
            model_kwargs["torch_dtype"] = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        # Store in cache
        loaded_models[model_path] = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "dtype": dtype
        }
        
        model_metadata[model_path] = {
            "loaded_at": datetime.now(),
            "last_used": datetime.now(),
            "ttl": ttl,
            "device": device,
            "torch_dtype": str(dtype)
        }
        
        logger.info(f"Model loaded successfully: {model_path}")
        return loaded_models[model_path]
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


async def unload_model(model_path: str) -> bool:
    """Unload a model from memory"""
    if model_path not in loaded_models:
        return False
    
    logger.info(f"Unloading model: {model_path}")
    
    try:
        # Get model info
        model_info = loaded_models[model_path]
        model = model_info["model"]
        
        # Move to CPU and delete
        if hasattr(model, 'cpu'):
            model.cpu()
        
        # Remove from cache
        del loaded_models[model_path]
        del model_metadata[model_path]
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
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
    logger.info("Transformers Engine Worker Starting")
    logger.info("=" * 50)
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Start cleanup task
    asyncio.create_task(cleanup_expired_models())


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine": "transformers",
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": len(loaded_models),
        "models": list(loaded_models.keys())
    }


@app.post("/v1/models/load")
async def load_model_endpoint(request: LoadModelRequest):
    """Load a model into memory"""
    try:
        await load_model_if_needed(
            model_path=request.model_path,
            device=request.device,
            torch_dtype=request.torch_dtype,
            ttl=request.ttl
        )
        
        return {
            "status": "loaded",
            "model_path": request.model_path,
            "device": request.device,
            "torch_dtype": request.torch_dtype
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
        model_info = await load_model_if_needed(
            model_path=request.model,
            device=request.device or "cuda",
            torch_dtype=request.torch_dtype or "auto",
            ttl=request.ttl or 300
        )
        
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        device = model_info["device"]
        
        # Convert messages to prompt
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        prompt = messages_to_prompt(messages_dict)
        
        logger.info(f"Generating with Transformers: max_tokens={request.max_tokens}, temperature={request.temperature}")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move to device
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        prompt_tokens = inputs.input_ids.shape[1]
        
        if request.stream:
            # Streaming response
            async def generate_stream():
                chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
                
                # First chunk with role
                first_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(first_chunk)}\n\n"
                
                # Create streamer
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                
                # Generation config
                generation_kwargs = {
                    "input_ids": inputs.input_ids,
                    "max_new_tokens": request.max_tokens,
                    "temperature": request.temperature if request.temperature > 0 else 1.0,
                    "top_p": request.top_p,
                    "do_sample": request.temperature > 0,
                    "streamer": streamer,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id
                }
                
                # Start generation in background thread
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Stream tokens
                try:
                    for text in streamer:
                        chunk_data = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                
                # Wait for thread
                thread.join()
                
                # Final chunk
                final_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
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
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature if request.temperature > 0 else 1.0,
                    top_p=request.top_p,
                    do_sample=request.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove prompt)
            generated_text = full_response[len(prompt):].strip()
            
            # Calculate token counts
            completion_tokens = outputs.shape[1] - prompt_tokens
            
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
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens + completion_tokens)
                }
            }
            
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Transformers worker on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

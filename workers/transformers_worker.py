from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import argparse
import time
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Transformers CPU Worker")

# Global model and tokenizer
model = None
tokenizer = None
model_name = None

def load_model(model_path: str):
    """Load model and tokenizer from disk (CPU only)"""
    global model, tokenizer, model_name
    
    logger.info(f"Loading model from {model_path} on CPU...")
    start_time = time.time()
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # Load model on CPU with float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        
        model_name = model_path.split("/")[-1]
        
        elapsed = time.time() - start_time
        logger.info(f"Model loaded successfully in {elapsed:.2f}s")
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: CPU")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

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
    
    # Add final assistant prompt
    prompt_parts.append("Assistant:")
    
    return "\n\n".join(prompt_parts)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Transformers CPU Worker",
        "model": model_name,
        "device": "cpu"
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    Generates responses using the loaded model on CPU.
    """
    if model is None or tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "Model not loaded",
                    "type": "service_unavailable"
                }
            }
        )
    
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid JSON: {str(e)}", "type": "invalid_request_error"}}
        )
    
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.7)
    
    if not messages:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Missing 'messages' field", "type": "invalid_request_error"}}
        )
    
    try:
        # Convert messages to prompt
        prompt = messages_to_prompt(messages)
        logger.info(f"Generating response for prompt (length: {len(prompt)})")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        generated_text = full_response[len(prompt):].strip()
        
        # Return OpenAI-compatible response
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
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
                "prompt_tokens": inputs.input_ids.shape[1],
                "completion_tokens": outputs.shape[1] - inputs.input_ids.shape[1],
                "total_tokens": outputs.shape[1]
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"Generation failed: {str(e)}",
                    "type": "internal_error"
                }
            }
        )

if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    # Load model at startup
    load_model(args.model_path)
    
    # Start server
    uvicorn.run(app, host=args.host, port=args.port)

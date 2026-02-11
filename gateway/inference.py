import logging
import time
import torch
from typing import List, Dict, Any, Optional

from models import BackendType, ChatMessage, ChatCompletionResponse, ChatChoice, UsageInfo, ChatCompletionChunk, ChatCompletionChunkChoice
from model_loader import ModelWrapper

logger = logging.getLogger(__name__)


def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Convert OpenAI message format to a prompt string
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        
    Returns:
        Formatted prompt string
    """
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


class InferenceEngine:
    """Unified inference engine for vLLM and Transformers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def generate(
        self,
        model_wrapper: ModelWrapper,
        messages: List[Dict[str, str]],
        params: Dict[str, Any]
    ) -> ChatCompletionResponse:
        """
        Generate response using the appropriate backend
        
        Args:
            model_wrapper: Loaded model wrapper
            messages: Chat messages
            params: Generation parameters
            
        Returns:
            OpenAI-compatible chat completion response
        """
        if model_wrapper.backend == BackendType.VLLM:
            return await self._generate_vllm(model_wrapper, messages, params)
        elif model_wrapper.backend == BackendType.TRANSFORMERS:
            return await self._generate_transformers(model_wrapper, messages, params)
        elif model_wrapper.backend == BackendType.LLAMACPP:
            return await self._generate_llamacpp(model_wrapper, messages, params)
        else:
            raise ValueError(f"Unsupported backend: {model_wrapper.backend}")
    
    async def generate_stream(
        self,
        model_wrapper: ModelWrapper,
        messages: List[Dict[str, str]],
        params: Dict[str, Any]
    ):
        """
        Generate streaming response using the appropriate backend
        
        Yields ChatCompletionChunk objects in SSE format
        """
        if model_wrapper.backend == BackendType.VLLM:
            async for chunk in self._generate_vllm_stream(model_wrapper, messages, params):
                yield chunk
        elif model_wrapper.backend == BackendType.TRANSFORMERS:
            async for chunk in self._generate_transformers_stream(model_wrapper, messages, params):
                yield chunk
        elif model_wrapper.backend == BackendType.LLAMACPP:
            async for chunk in self._generate_llamacpp_stream(model_wrapper, messages, params):
                yield chunk
        else:
            raise ValueError(f"Unsupported backend: {model_wrapper.backend}")
    
    async def _generate_vllm(
        self,
        model_wrapper: ModelWrapper,
        messages: List[Dict[str, str]],
        params: Dict[str, Any]
    ) -> ChatCompletionResponse:
        """
        Generate response using vLLM
        
        Args:
            model_wrapper: ModelWrapper with vLLM model
            messages: Chat messages
            params: Generation parameters
            
        Returns:
            ChatCompletionResponse
        """
        try:
            from vllm import SamplingParams
        except ImportError:
            raise RuntimeError("vLLM not installed")
        
        # Convert messages to prompt
        prompt = messages_to_prompt(messages)
        
        # Extract parameters
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 512)
        top_p = params.get("top_p", 1.0)
        stop = params.get("stop", None)
        presence_penalty = params.get("presence_penalty", 0.0)
        frequency_penalty = params.get("frequency_penalty", 0.0)
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty
        )
        
        self.logger.info(f"Generating with vLLM: max_tokens={max_tokens}, temperature={temperature}")
        
        # Generate (synchronous call, run in executor to not block)
        import asyncio
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: model_wrapper.model.generate([prompt], sampling_params)
        )
        
        # Extract output
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        # Estimate token counts
        prompt_tokens = len(prompt.split())  # Rough estimate
        completion_tokens = len(generated_text.split())  # Rough estimate
        
        # Format as OpenAI response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            object="chat.completion",
            created=int(time.time()),
            model=model_wrapper.model_path,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=generated_text
                    ),
                    finish_reason=output.outputs[0].finish_reason or "stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        self.logger.info(f"vLLM generation complete: {completion_tokens} tokens")
        
        return response
    
    async def _generate_vllm_stream(
        self,
        model_wrapper: ModelWrapper,
        messages: List[Dict[str, str]],
        params: Dict[str, Any]
    ):
        """Generate streaming response using vLLM"""
        from vllm import SamplingParams
        
        prompt = messages_to_prompt(messages)
        
        sampling_params = SamplingParams(
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 512),
            top_p=params.get("top_p", 1.0),
            stop=params.get("stop", None),
            presence_penalty=params.get("presence_penalty", 0.0),
            frequency_penalty=params.get("frequency_penalty", 0.0)
        )
        
        chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
        
        # First chunk with role
        yield ChatCompletionChunk(
            id=chunk_id,
            created=int(time.time()),
            model=model_wrapper.model_path,
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta={"role": "assistant"},
                finish_reason=None
            )]
        )
        
        # vLLM async streaming
        import asyncio
        async def stream_results():
            loop = asyncio.get_event_loop()
            results_generator = await loop.run_in_executor(
                None,
                lambda: model_wrapper.model.generate([prompt], sampling_params, use_tqdm=False)
            )
            for output in results_generator:
                yield output
        
        async for output in stream_results():
            text = output.outputs[0].text
            
            yield ChatCompletionChunk(
                id=chunk_id,
                created=int(time.time()),
                model=model_wrapper.model_path,
                choices=[ChatCompletionChunkChoice(
                    index=0,
                    delta={"content": text},
                    finish_reason=None
                )]
            )
        
        # Final chunk with finish_reason
        yield ChatCompletionChunk(
            id=chunk_id,
            created=int(time.time()),
            model=model_wrapper.model_path,
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta={},
                finish_reason="stop"
            )]
        )
    
    async def _generate_transformers(
        self,
        model_wrapper: ModelWrapper,
        messages: List[Dict[str, str]],
        params: Dict[str, Any]
    ) -> ChatCompletionResponse:
        """
        Generate response using Transformers
        
        Args:
            model_wrapper: ModelWrapper with Transformers model
            messages: Chat messages
            params: Generation parameters
            
        Returns:
            ChatCompletionResponse
        """
        # Convert messages to prompt
        prompt = messages_to_prompt(messages)
        
        # Extract parameters
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 512)
        top_p = params.get("top_p", 1.0)
        
        self.logger.info(f"Generating with Transformers: max_tokens={max_tokens}, temperature={temperature}")
        
        # Tokenize input
        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move to same device as model
        if model_wrapper.device.value == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        prompt_tokens = inputs.input_ids.shape[1]
        
        # Generate (synchronous call, run in executor)
        import asyncio
        loop = asyncio.get_event_loop()
        
        def generate_sync():
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            return outputs
        
        outputs = await loop.run_in_executor(None, generate_sync)
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove prompt)
        generated_text = full_response[len(prompt):].strip()
        
        # Calculate token counts
        completion_tokens = outputs.shape[1] - prompt_tokens
        
        # Format as OpenAI response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            object="chat.completion",
            created=int(time.time()),
            model=model_wrapper.model_path,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=generated_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(prompt_tokens + completion_tokens)
            )
        )
        
        self.logger.info(f"Transformers generation complete: {completion_tokens} tokens")
        
        return response
    
    async def _generate_transformers_stream(
        self,
        model_wrapper: ModelWrapper,
        messages: List[Dict[str, str]],
        params: Dict[str, Any]
    ):
        """Generate streaming response using Transformers with TextIteratorStreamer"""
        from transformers import TextIteratorStreamer
        from threading import Thread
        import asyncio
        
        prompt = messages_to_prompt(messages)
        
        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        if model_wrapper.device.value == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Create streamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generation config
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_new_tokens": params.get("max_tokens", 512),
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 1.0),
            "do_sample": params.get("temperature", 0.7) > 0,
            "streamer": streamer,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
        
        # Start generation in background thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
        
        # First chunk with role
        yield ChatCompletionChunk(
            id=chunk_id,
            created=int(time.time()),
            model=model_wrapper.model_path,
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta={"role": "assistant"},
                finish_reason=None
            )]
        )
        
        # Stream tokens
        loop = asyncio.get_event_loop()
        for text in streamer:
            await asyncio.sleep(0)  # Yield control to event loop
            
            yield ChatCompletionChunk(
                id=chunk_id,
                created=int(time.time()),
                model=model_wrapper.model_path,
                choices=[ChatCompletionChunkChoice(
                    index=0,
                    delta={"content": text},
                    finish_reason=None
                )]
            )
        
        # Wait for thread to complete
        await loop.run_in_executor(None, thread.join)
        
        # Final chunk
        yield ChatCompletionChunk(
            id=chunk_id,
            created=int(time.time()),
            model=model_wrapper.model_path,
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta={},
                finish_reason="stop"
            )]
        )
    
    async def _generate_llamacpp(
        self,
        model_wrapper: ModelWrapper,
        messages: List[Dict[str, str]],
        params: Dict[str, Any]
    ) -> ChatCompletionResponse:
        """
        Generate response using llama.cpp
        
        Args:
            model_wrapper: ModelWrapper with llama.cpp model
            messages: Chat messages
            params: Generation parameters
            
        Returns:
            ChatCompletionResponse
        """
        # Convert messages to prompt
        prompt = messages_to_prompt(messages)
        
        # Extract parameters
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 512)
        top_p = params.get("top_p", 1.0)
        stop = params.get("stop", None)
        
        self.logger.info(f"Generating with llama.cpp: max_tokens={max_tokens}, temperature={temperature}")
        
        # Generate (synchronous call, run in executor)
        import asyncio
        loop = asyncio.get_event_loop()
        
        def generate_sync():
            output = model_wrapper.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop if stop else [],
                echo=False
            )
            return output
        
        output = await loop.run_in_executor(None, generate_sync)
        
        # Extract generated text
        generated_text = output['choices'][0]['text']
        
        # Get token counts
        prompt_tokens = output['usage']['prompt_tokens']
        completion_tokens = output['usage']['completion_tokens']
        
        # Format as OpenAI response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            object="chat.completion",
            created=int(time.time()),
            model=model_wrapper.model_path,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=generated_text.strip()
                    ),
                    finish_reason=output['choices'][0].get('finish_reason', 'stop')
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        self.logger.info(f"llama.cpp generation complete: {completion_tokens} tokens")
        
        return response
    
    async def _generate_llamacpp_stream(
        self,
        model_wrapper: ModelWrapper,
        messages: List[Dict[str, str]],
        params: Dict[str, Any]
    ):
        """Generate streaming response using llama.cpp"""
        import asyncio
        
        prompt = messages_to_prompt(messages)
        
        chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
        
        # First chunk with role
        yield ChatCompletionChunk(
            id=chunk_id,
            created=int(time.time()),
            model=model_wrapper.model_path,
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta={"role": "assistant"},
                finish_reason=None
            )]
        )
        
        # llama.cpp streaming
        loop = asyncio.get_event_loop()
        
        def generate_sync():
            return model_wrapper.model(
                prompt,
                max_tokens=params.get("max_tokens", 512),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 1.0),
                stop=params.get("stop", []),
                echo=False,
                stream=True  # Enable streaming
            )
        
        stream = await loop.run_in_executor(None, generate_sync)
        
        # Iterate over streaming tokens
        for chunk in stream:
            await asyncio.sleep(0)  # Yield control
            
            text = chunk['choices'][0].get('text', '')
            finish_reason = chunk['choices'][0].get('finish_reason', None)
            
            yield ChatCompletionChunk(
                id=chunk_id,
                created=int(time.time()),
                model=model_wrapper.model_path,
                choices=[ChatCompletionChunkChoice(
                    index=0,
                    delta={"content": text} if text else {},
                    finish_reason=finish_reason
                )]
            )
            
            if finish_reason:
                break


# Global inference engine instance
inference_engine = InferenceEngine()

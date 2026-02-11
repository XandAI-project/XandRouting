from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, Dict, Any, List
from datetime import datetime
from enum import Enum


class BackendType(str, Enum):
    """Supported backend types"""
    VLLM = "vllm"
    TRANSFORMERS = "transformers"
    LLAMACPP = "llamacpp"


class DeviceType(str, Enum):
    """Supported device types"""
    CUDA = "cuda"
    CPU = "cpu"


class ModelConfig(BaseModel):
    """Configuration for loading a model"""
    model: str = Field(..., description="Path to model directory or GGUF file, e.g., /models/gpt-oss or /models/model.gguf")
    backend: BackendType = Field(..., description="Backend to use: vllm, transformers, or llamacpp")
    device: DeviceType = Field(..., description="Device to run on: cuda or cpu")
    gpu_memory_utilization: float = Field(default=0.4, ge=0.1, le=1.0, description="GPU memory utilization (vLLM only)")
    ttl: int = Field(default=300, ge=0, description="Time-to-live in seconds, 0 = no expiry")
    n_gpu_layers: int = Field(default=-1, description="Number of layers to offload to GPU (llama.cpp only, -1 = all)")
    n_ctx: int = Field(default=4096, description="Context window size (llama.cpp only)")
    max_model_len: Optional[int] = Field(default=None, description="Maximum model context length (vLLM only)")
    
    @validator('backend')
    def validate_backend(cls, v):
        """Ensure backend is valid"""
        if v not in [BackendType.VLLM, BackendType.TRANSFORMERS, BackendType.LLAMACPP]:
            raise ValueError(f"Backend must be 'vllm', 'transformers', or 'llamacpp', got: {v}")
        return v
    
    @validator('device')
    def validate_device(cls, v):
        """Ensure device is valid"""
        if v not in [DeviceType.CUDA, DeviceType.CPU]:
            raise ValueError(f"Device must be 'cuda' or 'cpu', got: {v}")
        return v
    
    def generate_cache_key(self) -> str:
        """Generate unique cache key for this configuration"""
        return f"{self.model}:{self.backend.value}:{self.device.value}"


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with dynamic model config"""
    # Model configuration (dynamic)
    model: str = Field(..., description="Path to model directory or GGUF file")
    backend: BackendType = Field(..., description="Backend: vllm, transformers, or llamacpp")
    device: DeviceType = Field(..., description="Device: cuda or cpu")
    gpu_memory_utilization: Optional[float] = Field(default=0.4, ge=0.1, le=1.0)
    ttl: Optional[int] = Field(default=300, ge=0)
    n_gpu_layers: Optional[int] = Field(default=-1, description="GPU layers for llama.cpp")
    n_ctx: Optional[int] = Field(default=4096, description="Context size for llama.cpp")
    max_model_len: Optional[int] = Field(default=None, description="Max context length for vLLM")
    
    # OpenAI parameters
    messages: list = Field(..., description="Array of message objects")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = Field(default=False)
    stop: Optional[list[str]] = None
    max_tokens: Optional[int] = Field(default=512, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ChatMessage(BaseModel):
    """Chat message"""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatChoice(BaseModel):
    """Chat completion choice"""
    index: int
    message: ChatMessage
    finish_reason: str


class UsageInfo(BaseModel):
    """Token usage information"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo


class ChatCompletionChunkChoice(BaseModel):
    """Streaming chunk choice"""
    index: int
    delta: Dict[str, Any]  # Contains 'role' and/or 'content'
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class CachedModel(BaseModel):
    """Information about a cached model"""
    cache_key: str
    model_path: str
    backend: BackendType
    device: DeviceType
    gpu_memory_utilization: float
    loaded_at: datetime
    last_used: datetime
    expires_at: Optional[datetime]
    ttl: int
    ttl_remaining: Optional[int] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LoadedModelsResponse(BaseModel):
    """Response for listing loaded models"""
    loaded_models: list[CachedModel]
    total_count: int


class ModelStatsResponse(BaseModel):
    """Statistics about model cache"""
    total_models_loaded: int
    active_models: int
    total_requests_served: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    estimated_gpu_memory_mb: Optional[float] = None
    estimated_cpu_memory_mb: Optional[float] = None


class UnloadModelRequest(BaseModel):
    """Request to unload a specific model"""
    model: str
    backend: Optional[BackendType] = None
    device: Optional[DeviceType] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: Dict[str, Any]


class ModelFile(BaseModel):
    """Information about a model file"""
    path: str
    filename: str
    size_bytes: int
    size_mb: float
    size_gb: float
    file_type: str  # "gguf", "safetensors", "bin", "pt", etc.
    quantization: Optional[str] = None  # For GGUF: "Q4_K_M", "IQ4_XS", etc.


class DiscoveredModel(BaseModel):
    """Information about a discovered model on disk"""
    model_name: str
    model_path: str
    total_size_bytes: int
    total_size_gb: float
    file_count: int
    files: List[ModelFile]
    model_type: str  # "gguf", "transformers", "unknown"
    recommended_backend: List[str]  # ["llamacpp"] or ["vllm", "transformers"]
    has_config: bool
    config_metadata: Optional[Dict[str, Any]] = None


class ModelInventoryResponse(BaseModel):
    """Response for model inventory endpoint"""
    models: List[DiscoveredModel]
    total_models: int
    total_size_gb: float

import logging
import torch
from typing import Union, Dict, Any
from pathlib import Path

from models import ModelConfig, BackendType, DeviceType

logger = logging.getLogger(__name__)


class ModelWrapper:
    """Wrapper for loaded models to provide unified interface"""
    
    def __init__(self, model: Any, config: ModelConfig, tokenizer: Any = None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.backend = config.backend
        self.device = config.device
        self.model_path = config.model
        
    def get_backend_type(self) -> str:
        """Get backend type as string"""
        return self.backend.value if hasattr(self.backend, 'value') else str(self.backend)


class ModelLoader:
    """Dynamically loads models based on backend configuration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, config: ModelConfig) -> ModelWrapper:
        """
        Load model based on configuration
        
        Args:
            config: ModelConfig specifying backend, device, and model path
            
        Returns:
            ModelWrapper containing the loaded model
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If model path doesn't exist
            RuntimeError: If model loading fails
        """
        # Validate model path exists
        model_path = Path(config.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {config.model}")
        
        self.logger.info(f"Loading model: {config.model} with backend={config.backend.value}, device={config.device.value}")
        
        try:
            if config.backend == BackendType.VLLM:
                return self._load_vllm(config)
            elif config.backend == BackendType.TRANSFORMERS:
                return self._load_transformers(config)
            elif config.backend == BackendType.LLAMACPP:
                return self._load_llamacpp(config)
            else:
                raise ValueError(f"Unsupported backend: {config.backend}")
        except Exception as e:
            self.logger.error(f"Failed to load model {config.model}: {e}", exc_info=True)
            raise RuntimeError(f"Model loading failed: {str(e)}") from e
    
    def _load_vllm(self, config: ModelConfig) -> ModelWrapper:
        """
        Load model using vLLM
        
        Args:
            config: ModelConfig with vLLM settings
            
        Returns:
            ModelWrapper with vLLM model
        """
        try:
            from vllm import LLM
        except ImportError:
            raise RuntimeError("vLLM is not installed. Install with: pip install vllm")
        
        # vLLM requires CUDA
        if config.device == DeviceType.CPU:
            self.logger.warning("vLLM requires CUDA, but CPU specified. Attempting to use CUDA anyway.")
            # vLLM will fail if no CUDA available, which is fine
        
        self.logger.info(f"Initializing vLLM with gpu_memory_utilization={config.gpu_memory_utilization}")
        
        # Build vLLM parameters
        vllm_params = {
            "model": config.model,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "trust_remote_code": True,
            "download_dir": None,
            "dtype": "auto"
        }
        
        # Add max_model_len if specified to bypass rope_scaling validation
        if config.max_model_len is not None:
            vllm_params["max_model_len"] = config.max_model_len
            self.logger.info(f"Using explicit max_model_len: {config.max_model_len}")
        
        # Load model with vLLM
        llm = LLM(**vllm_params)
        
        self.logger.info(f"Successfully loaded vLLM model: {config.model}")
        
        return ModelWrapper(model=llm, config=config)
    
    def _load_transformers(self, config: ModelConfig) -> ModelWrapper:
        """
        Load model using HuggingFace Transformers
        
        Args:
            config: ModelConfig with Transformers settings
            
        Returns:
            ModelWrapper with Transformers model and tokenizer
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise RuntimeError("Transformers is not installed. Install with: pip install transformers")
        
        self.logger.info(f"Loading tokenizer from {config.model}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            trust_remote_code=True,
            local_files_only=False
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.logger.info(f"Loading model from {config.model}")
        
        # Determine dtype based on device
        if config.device == DeviceType.CUDA:
            torch_dtype = torch.float16
            device_map = "auto"
        else:
            torch_dtype = torch.float32
            device_map = "cpu"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=False,
            low_cpu_mem_usage=True
        )
        
        # Set model to evaluation mode
        model.eval()
        
        self.logger.info(f"Successfully loaded Transformers model: {config.model}")
        
        return ModelWrapper(model=model, config=config, tokenizer=tokenizer)
    
    def _load_llamacpp(self, config: ModelConfig) -> ModelWrapper:
        """
        Load GGUF model using llama.cpp
        
        Args:
            config: ModelConfig with llama.cpp settings
            
        Returns:
            ModelWrapper with llama.cpp model
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError("llama-cpp-python is not installed. Install with: pip install llama-cpp-python")
        
        # Validate GGUF file
        model_path = Path(config.model)
        if model_path.is_file() and not config.model.endswith('.gguf'):
            self.logger.warning(f"File {config.model} doesn't have .gguf extension, but attempting to load anyway")
        
        # Determine GPU layers
        n_gpu_layers = config.n_gpu_layers if hasattr(config, 'n_gpu_layers') else -1
        if config.device == DeviceType.CPU:
            n_gpu_layers = 0
            self.logger.info("CPU mode requested, setting n_gpu_layers=0")
        
        n_ctx = config.n_ctx if hasattr(config, 'n_ctx') else 2048
        
        self.logger.info(f"Initializing llama.cpp with n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}")
        
        # Load model with llama.cpp
        llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        
        self.logger.info(f"Successfully loaded llama.cpp model: {config.model}")
        
        return ModelWrapper(model=llm, config=config)
    
    def unload_model(self, model_wrapper: ModelWrapper) -> None:
        """
        Unload model and free memory
        
        Args:
            model_wrapper: ModelWrapper to unload
        """
        self.logger.info(f"Unloading model: {model_wrapper.model_path}")
        
        try:
            if model_wrapper.backend == BackendType.VLLM:
                # vLLM cleanup
                if hasattr(model_wrapper.model, 'llm_engine'):
                    del model_wrapper.model.llm_engine
                del model_wrapper.model
            elif model_wrapper.backend == BackendType.TRANSFORMERS:
                # Transformers cleanup
                if model_wrapper.model is not None:
                    del model_wrapper.model
                if model_wrapper.tokenizer is not None:
                    del model_wrapper.tokenizer
            elif model_wrapper.backend == BackendType.LLAMACPP:
                # llama.cpp cleanup
                if model_wrapper.model is not None:
                    del model_wrapper.model
            
            # Force garbage collection and CUDA cache cleanup
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.logger.info(f"Successfully unloaded model: {model_wrapper.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error unloading model {model_wrapper.model_path}: {e}", exc_info=True)
    
    def estimate_model_memory(self, model_wrapper: ModelWrapper) -> Dict[str, float]:
        """
        Estimate memory usage of loaded model
        
        Args:
            model_wrapper: ModelWrapper to estimate
            
        Returns:
            Dict with 'gpu_mb' and 'cpu_mb' estimates
        """
        gpu_mb = 0.0
        cpu_mb = 0.0
        
        try:
            if model_wrapper.backend == BackendType.VLLM:
                # vLLM memory estimation
                if torch.cuda.is_available():
                    gpu_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            
            elif model_wrapper.backend == BackendType.LLAMACPP:
                # llama.cpp memory estimation (rough)
                if model_wrapper.device == DeviceType.CUDA and torch.cuda.is_available():
                    gpu_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                else:
                    # Estimate CPU memory based on file size
                    try:
                        file_size = Path(model_wrapper.model_path).stat().st_size
                        cpu_mb = file_size / (1024 ** 2)
                    except:
                        cpu_mb = 0
            
            elif model_wrapper.backend == BackendType.TRANSFORMERS:
                # Transformers memory estimation
                if hasattr(model_wrapper.model, 'get_memory_footprint'):
                    total_bytes = model_wrapper.model.get_memory_footprint()
                    if model_wrapper.device == DeviceType.CUDA:
                        gpu_mb = total_bytes / (1024 ** 2)
                    else:
                        cpu_mb = total_bytes / (1024 ** 2)
                else:
                    # Fallback estimation
                    param_count = sum(p.numel() for p in model_wrapper.model.parameters())
                    bytes_per_param = 2  # fp16
                    total_mb = (param_count * bytes_per_param) / (1024 ** 2)
                    
                    if model_wrapper.device == DeviceType.CUDA:
                        gpu_mb = total_mb
                    else:
                        cpu_mb = total_mb
        
        except Exception as e:
            self.logger.warning(f"Could not estimate memory for {model_wrapper.model_path}: {e}")
        
        return {"gpu_mb": gpu_mb, "cpu_mb": cpu_mb}

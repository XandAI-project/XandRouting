"""
Model Scanner Utility

Scans the /models/ directory to discover all downloaded models,
detect their types, and recommend appropriate backends.
"""

import os
import json
import re
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from models import ModelFile, DiscoveredModel

logger = logging.getLogger(__name__)


def extract_quantization_from_filename(filename: str) -> Optional[str]:
    """
    Extract quantization type from GGUF filename
    
    Examples:
    - model-Q4_K_M.gguf -> Q4_K_M
    - mistral-7b-IQ4_XS.gguf -> IQ4_XS
    - llama-Q8_0.gguf -> Q8_0
    
    Supports common quantization types:
    - Q2_K, Q3_K_S, Q3_K_M, Q3_K_L
    - Q4_0, Q4_1, Q4_K_S, Q4_K_M
    - Q5_0, Q5_1, Q5_K_S, Q5_K_M
    - Q6_K, Q8_0
    - IQ1_S, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_XS, IQ4_XS, IQ4_NL
    """
    if not filename.endswith('.gguf'):
        return None
    
    # Pattern to match quantization types
    pattern = r'[-_](Q\d+_K_[SML]|Q\d+_[01]|IQ\d+_[A-Z]+)\.gguf$'
    match = re.search(pattern, filename, re.IGNORECASE)
    
    if match:
        return match.group(1).upper()
    
    return None


def get_file_type(filename: str) -> str:
    """
    Determine file type from extension
    """
    ext = filename.lower().split('.')[-1]
    
    file_type_mapping = {
        'gguf': 'gguf',
        'safetensors': 'safetensors',
        'bin': 'bin',
        'pt': 'pt',
        'pth': 'pth',
        'json': 'json',
        'txt': 'txt',
        'md': 'markdown',
        'py': 'python'
    }
    
    return file_type_mapping.get(ext, ext)


def calculate_directory_size(directory: str) -> int:
    """
    Recursively calculate total size of all files in directory
    
    Returns size in bytes
    """
    total_size = 0
    
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
                except OSError as e:
                    logger.warning(f"Could not get size of {filepath}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error calculating directory size for {directory}: {e}")
    
    return total_size


def read_model_config(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Read and parse config.json if it exists
    
    Returns metadata dict or None if not found/invalid
    """
    config_path = os.path.join(model_path, "config.json")
    
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Extract relevant metadata
        metadata = {}
        
        # Common fields
        if 'model_type' in config:
            metadata['model_type'] = config['model_type']
        if 'architectures' in config:
            metadata['architectures'] = config['architectures']
        if 'num_hidden_layers' in config:
            metadata['num_hidden_layers'] = config['num_hidden_layers']
        if 'hidden_size' in config:
            metadata['hidden_size'] = config['hidden_size']
        if 'vocab_size' in config:
            metadata['vocab_size'] = config['vocab_size']
        if 'max_position_embeddings' in config:
            metadata['max_position_embeddings'] = config['max_position_embeddings']
        
        return metadata if metadata else None
        
    except Exception as e:
        logger.warning(f"Could not read config.json from {config_path}: {e}")
        return None


def detect_model_type(model_path: str, files: List[str]) -> str:
    """
    Detect model type based on files present
    
    Returns: "gguf", "transformers", or "unknown"
    """
    # Check for GGUF files
    has_gguf = any(f.endswith('.gguf') for f in files)
    if has_gguf:
        return "gguf"
    
    # Check for transformers model files
    has_config = "config.json" in files
    has_safetensors = any(f.endswith('.safetensors') for f in files)
    has_bin = any(f.endswith('.bin') for f in files)
    has_pt = any(f.endswith('.pt') or f.endswith('.pth') for f in files)
    
    if has_config and (has_safetensors or has_bin or has_pt):
        return "transformers"
    
    return "unknown"


def recommend_backend(model_type: str, config_metadata: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Recommend backend(s) based on model type and metadata
    
    Returns list of recommended backends in priority order
    """
    if model_type == "gguf":
        # GGUF files are only compatible with llama.cpp
        return ["llamacpp"]
    
    elif model_type == "transformers":
        # Transformers models work with both vLLM and Transformers
        # vLLM is preferred for better performance but not all models support it
        return ["vllm", "transformers"]
    
    else:
        # Unknown type - default to Transformers as most flexible
        return ["transformers"]


def scan_model_directory(model_path: str) -> Optional[DiscoveredModel]:
    """
    Scan a single model directory and extract all information
    
    Returns DiscoveredModel or None if invalid
    """
    try:
        if not os.path.isdir(model_path):
            return None
        
        model_name = os.path.basename(model_path)
        
        # Get all files (non-recursive)
        all_files = []
        try:
            all_files = os.listdir(model_path)
        except PermissionError:
            logger.warning(f"Permission denied accessing {model_path}")
            return None
        
        if not all_files:
            return None
        
        # Detect model type
        model_type = detect_model_type(model_path, all_files)
        
        # Read config if available
        config_metadata = read_model_config(model_path)
        has_config = config_metadata is not None
        
        # Recommend backend
        recommended_backend = recommend_backend(model_type, config_metadata)
        
        # Process each file
        model_files: List[ModelFile] = []
        total_size_bytes = 0
        
        for filename in all_files:
            filepath = os.path.join(model_path, filename)
            
            # Skip directories
            if os.path.isdir(filepath):
                continue
            
            # Skip hidden files and common non-model files
            if filename.startswith('.') or filename in ['README.md', 'LICENSE', '.gitattributes']:
                continue
            
            try:
                size_bytes = os.path.getsize(filepath)
                total_size_bytes += size_bytes
                
                size_mb = size_bytes / (1024 * 1024)
                size_gb = size_bytes / (1024 * 1024 * 1024)
                
                file_type = get_file_type(filename)
                quantization = extract_quantization_from_filename(filename)
                
                model_file = ModelFile(
                    path=filepath,
                    filename=filename,
                    size_bytes=size_bytes,
                    size_mb=round(size_mb, 2),
                    size_gb=round(size_gb, 2),
                    file_type=file_type,
                    quantization=quantization
                )
                
                model_files.append(model_file)
                
            except OSError as e:
                logger.warning(f"Could not process file {filepath}: {e}")
                continue
        
        if not model_files:
            return None
        
        total_size_gb = round(total_size_bytes / (1024 * 1024 * 1024), 2)
        
        discovered_model = DiscoveredModel(
            model_name=model_name,
            model_path=model_path,
            total_size_bytes=total_size_bytes,
            total_size_gb=total_size_gb,
            file_count=len(model_files),
            files=model_files,
            model_type=model_type,
            recommended_backend=recommended_backend,
            has_config=has_config,
            config_metadata=config_metadata
        )
        
        return discovered_model
        
    except Exception as e:
        logger.error(f"Error scanning model directory {model_path}: {e}", exc_info=True)
        return None


async def scan_models_directory(models_dir: str) -> Dict[str, Any]:
    """
    Main function to scan all models in the /models/ directory
    
    Returns dict with:
    - models: List[DiscoveredModel]
    - total_size_gb: float
    """
    logger.info(f"Scanning models directory: {models_dir}")
    
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory does not exist: {models_dir}")
        return {
            "models": [],
            "total_size_gb": 0.0
        }
    
    if not os.path.isdir(models_dir):
        logger.error(f"Models path is not a directory: {models_dir}")
        return {
            "models": [],
            "total_size_gb": 0.0
        }
    
    discovered_models: List[DiscoveredModel] = []
    total_size_bytes = 0
    
    try:
        # List all subdirectories (each should be a model)
        entries = os.listdir(models_dir)
        
        for entry in entries:
            entry_path = os.path.join(models_dir, entry)
            
            # Skip if not a directory
            if not os.path.isdir(entry_path):
                continue
            
            # Skip hidden directories
            if entry.startswith('.'):
                continue
            
            # Scan this model directory
            model = scan_model_directory(entry_path)
            
            if model:
                discovered_models.append(model)
                total_size_bytes += model.total_size_bytes
                logger.info(f"Discovered model: {model.model_name} ({model.total_size_gb} GB, type: {model.model_type})")
        
        total_size_gb = round(total_size_bytes / (1024 * 1024 * 1024), 2)
        
        logger.info(f"Scan complete: {len(discovered_models)} models, {total_size_gb} GB total")
        
        return {
            "models": discovered_models,
            "total_size_gb": total_size_gb
        }
        
    except Exception as e:
        logger.error(f"Error scanning models directory: {e}", exc_info=True)
        return {
            "models": [],
            "total_size_gb": 0.0
        }

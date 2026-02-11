import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from collections import OrderedDict

from models import ModelConfig, CachedModel, BackendType, DeviceType
from model_loader import ModelLoader, ModelWrapper

logger = logging.getLogger(__name__)


class ModelCacheEntry:
    """Entry in the model cache"""
    
    def __init__(self, model_wrapper: ModelWrapper, config: ModelConfig):
        self.model_wrapper = model_wrapper
        self.config = config
        self.cache_key = config.generate_cache_key()
        self.loaded_at = datetime.now()
        self.last_used = datetime.now()
        self.ttl = config.ttl
        self.access_count = 0
        
    def update_last_used(self):
        """Update last used timestamp"""
        self.last_used = datetime.now()
        self.access_count += 1
    
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL"""
        if self.ttl == 0:
            return False  # No expiry
        
        expiry_time = self.last_used + timedelta(seconds=self.ttl)
        return datetime.now() > expiry_time
    
    def get_expires_at(self) -> Optional[datetime]:
        """Get expiration datetime"""
        if self.ttl == 0:
            return None
        return self.last_used + timedelta(seconds=self.ttl)
    
    def get_ttl_remaining(self) -> Optional[int]:
        """Get remaining TTL in seconds"""
        if self.ttl == 0:
            return None
        
        expires_at = self.get_expires_at()
        if expires_at:
            remaining = (expires_at - datetime.now()).total_seconds()
            return max(0, int(remaining))
        return None
    
    def to_cached_model(self) -> CachedModel:
        """Convert to CachedModel for API responses"""
        return CachedModel(
            cache_key=self.cache_key,
            model_path=self.config.model,
            backend=self.config.backend,
            device=self.config.device,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            loaded_at=self.loaded_at,
            last_used=self.last_used,
            expires_at=self.get_expires_at(),
            ttl=self.ttl,
            ttl_remaining=self.get_ttl_remaining()
        )


class ModelCache:
    """
    Thread-safe cache for loaded models with TTL-based expiration
    Implements exclusive mode where same model can't run with different configs
    """
    
    def __init__(self, cleanup_interval: int = 30, max_gpu_memory_gb: Optional[float] = None):
        """
        Initialize model cache
        
        Args:
            cleanup_interval: Seconds between cleanup runs
            max_gpu_memory_gb: Maximum GPU memory to use (None = no limit)
        """
        self.cache: OrderedDict[str, ModelCacheEntry] = OrderedDict()
        self.locks: Dict[str, asyncio.Lock] = {}
        self.global_lock = threading.RLock()
        self.loader = ModelLoader()
        self.cleanup_interval = cleanup_interval
        self.max_gpu_memory_gb = max_gpu_memory_gb
        
        # Statistics
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"ModelCache initialized with cleanup_interval={cleanup_interval}s")
    
    async def get_or_load(self, config: ModelConfig) -> ModelWrapper:
        """
        Get model from cache or load it
        Thread-safe with per-cache-key locking
        Implements exclusive mode: same model_path can't exist with different config
        
        Args:
            config: ModelConfig specifying model and backend
            
        Returns:
            ModelWrapper for the loaded model
        """
        cache_key = config.generate_cache_key()
        self.total_requests += 1
        
        # Create lock for this cache key if it doesn't exist
        if cache_key not in self.locks:
            with self.global_lock:
                if cache_key not in self.locks:
                    self.locks[cache_key] = asyncio.Lock()
        
        # Acquire lock for this specific cache key
        async with self.locks[cache_key]:
            # Check if already in cache
            if cache_key in self.cache:
                logger.info(f"Cache HIT: {cache_key}")
                self.cache_hits += 1
                entry = self.cache[cache_key]
                entry.update_last_used()
                
                # Move to end (LRU)
                self.cache.move_to_end(cache_key)
                
                return entry.model_wrapper
            
            # Cache miss - need to load
            logger.info(f"Cache MISS: {cache_key}")
            self.cache_misses += 1
            
            # EXCLUSIVE MODE: Unload same model with different config
            self._unload_same_model_different_config(config)
            
            # Check if we need to evict models due to memory constraints
            if self.max_gpu_memory_gb:
                self._evict_if_needed()
            
            # Load the model
            logger.info(f"Loading model: {cache_key}")
            model_wrapper = await asyncio.get_event_loop().run_in_executor(
                None, self.loader.load_model, config
            )
            
            # Create cache entry
            entry = ModelCacheEntry(model_wrapper, config)
            
            # Add to cache
            with self.global_lock:
                self.cache[cache_key] = entry
            
            logger.info(f"Model loaded and cached: {cache_key}")
            
            return model_wrapper
    
    def _unload_same_model_different_config(self, config: ModelConfig):
        """
        Unload same model with different configuration (exclusive mode)
        
        Args:
            config: New configuration being loaded
        """
        model_path = config.model
        cache_key = config.generate_cache_key()
        
        with self.global_lock:
            keys_to_remove = []
            
            # Find same model with different config
            for key, entry in self.cache.items():
                if entry.config.model == model_path and key != cache_key:
                    logger.info(f"Exclusive mode: Unloading {key} to load {cache_key}")
                    keys_to_remove.append(key)
            
            # Unload found models
            for key in keys_to_remove:
                self._unload_entry(key)
    
    def _evict_if_needed(self):
        """Evict LRU models if approaching memory limit"""
        if not self.max_gpu_memory_gb:
            return
        
        # This is a simplified version - in production you'd track actual memory
        logger.debug("Checking if eviction needed...")
    
    def _unload_entry(self, cache_key: str):
        """
        Unload a specific cache entry
        
        Args:
            cache_key: Key of entry to unload
        """
        with self.global_lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                logger.info(f"Unloading model: {cache_key}")
                
                try:
                    self.loader.unload_model(entry.model_wrapper)
                except Exception as e:
                    logger.error(f"Error unloading {cache_key}: {e}")
                
                del self.cache[cache_key]
                
                # Clean up lock
                if cache_key in self.locks:
                    del self.locks[cache_key]
    
    def _cleanup_loop(self):
        """Background thread to clean up expired models"""
        logger.info("Cleanup thread started")
        
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)
    
    def _cleanup_expired(self):
        """Remove expired models from cache"""
        with self.global_lock:
            keys_to_remove = []
            
            for key, entry in self.cache.items():
                if entry.is_expired():
                    logger.info(f"Model expired: {key}")
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._unload_entry(key)
            
            if keys_to_remove:
                logger.info(f"Cleaned up {len(keys_to_remove)} expired models")
    
    def get_loaded_models(self) -> List[CachedModel]:
        """Get list of currently loaded models"""
        with self.global_lock:
            return [entry.to_cached_model() for entry in self.cache.values()]
    
    def get_model_count(self) -> int:
        """Get count of loaded models"""
        with self.global_lock:
            return len(self.cache)
    
    def unload_model(self, cache_key: str) -> bool:
        """
        Manually unload a specific model
        
        Args:
            cache_key: Cache key of model to unload
            
        Returns:
            True if unloaded, False if not found
        """
        with self.global_lock:
            if cache_key in self.cache:
                self._unload_entry(cache_key)
                return True
            return False
    
    def unload_by_path(self, model_path: str) -> int:
        """
        Unload all models with given model path
        
        Args:
            model_path: Model path to unload
            
        Returns:
            Number of models unloaded
        """
        with self.global_lock:
            keys_to_remove = [
                key for key, entry in self.cache.items()
                if entry.config.model == model_path
            ]
            
            for key in keys_to_remove:
                self._unload_entry(key)
            
            return len(keys_to_remove)
    
    def unload_all(self) -> int:
        """
        Unload all models
        
        Returns:
            Number of models unloaded
        """
        with self.global_lock:
            count = len(self.cache)
            keys = list(self.cache.keys())
            
            for key in keys:
                self._unload_entry(key)
            
            logger.info(f"Unloaded all {count} models")
            return count
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.global_lock:
            hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0.0
            
            # Estimate memory usage
            gpu_memory_mb = 0.0
            cpu_memory_mb = 0.0
            
            for entry in self.cache.values():
                memory = self.loader.estimate_model_memory(entry.model_wrapper)
                gpu_memory_mb += memory.get('gpu_mb', 0)
                cpu_memory_mb += memory.get('cpu_mb', 0)
            
            return {
                "total_models_loaded": len(self.cache),
                "active_models": len(self.cache),
                "total_requests_served": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": round(hit_rate, 2),
                "estimated_gpu_memory_mb": round(gpu_memory_mb, 2) if gpu_memory_mb > 0 else None,
                "estimated_cpu_memory_mb": round(cpu_memory_mb, 2) if cpu_memory_mb > 0 else None
            }


# Global cache instance
model_cache = ModelCache(cleanup_interval=30)

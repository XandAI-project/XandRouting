import os
import logging
import httpx
from typing import Dict, Optional
from models import BackendType

logger = logging.getLogger(__name__)


class EngineRouter:
    """
    Service discovery and routing for backend engine services
    Routes requests to appropriate worker services based on backend type
    """
    
    def __init__(self):
        """Initialize engine router with service endpoints from environment"""
        self.engines = {
            BackendType.VLLM: os.getenv("VLLM_SERVICE", "vllm-engine:8000"),
            BackendType.TRANSFORMERS: os.getenv("TRANSFORMERS_SERVICE", "transformers-engine:8000"),
            BackendType.LLAMACPP: os.getenv("LLAMACPP_SERVICE", "llamacpp-engine:8000")
        }
        
        logger.info("Engine Router initialized with services:")
        for backend, service in self.engines.items():
            logger.info(f"  {backend.value}: http://{service}")
    
    def get_engine_url(self, backend: BackendType) -> str:
        """
        Get full URL for a backend engine service
        
        Args:
            backend: Backend type (vllm, transformers, llamacpp)
            
        Returns:
            Full HTTP URL to the engine service
            
        Raises:
            ValueError: If backend type is not supported
        """
        if backend not in self.engines:
            raise ValueError(f"Unsupported backend: {backend}")
        
        service_host = self.engines[backend]
        return f"http://{service_host}"
    
    async def health_check(self, backend: BackendType) -> Dict:
        """
        Check health of a specific engine service
        
        Args:
            backend: Backend type to check
            
        Returns:
            Health check response dict
        """
        try:
            url = self.get_engine_url(backend)
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/health")
                response.raise_for_status()
                return {
                    "backend": backend.value,
                    "status": "healthy",
                    "url": url,
                    "details": response.json()
                }
        except Exception as e:
            logger.error(f"Health check failed for {backend.value}: {e}")
            return {
                "backend": backend.value,
                "status": "unhealthy",
                "url": url if 'url' in locals() else "unknown",
                "error": str(e)
            }
    
    async def health_check_all(self) -> Dict[str, Dict]:
        """
        Check health of all engine services
        
        Returns:
            Dict mapping backend name to health status
        """
        results = {}
        
        for backend in self.engines.keys():
            results[backend.value] = await self.health_check(backend)
        
        return results
    
    async def proxy_request(
        self,
        backend: BackendType,
        endpoint: str,
        method: str = "POST",
        json_data: Optional[Dict] = None,
        timeout: float = 300.0
    ) -> httpx.Response:
        """
        Proxy a request to a backend engine service
        
        Args:
            backend: Backend type to route to
            endpoint: API endpoint (e.g., "/v1/chat/completions")
            method: HTTP method (GET, POST, etc.)
            json_data: JSON request body
            timeout: Request timeout in seconds
            
        Returns:
            httpx.Response object
            
        Raises:
            httpx.HTTPError: If request fails
        """
        url = self.get_engine_url(backend)
        full_url = f"{url}{endpoint}"
        
        logger.info(f"Proxying {method} request to {full_url}")
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method.upper() == "POST":
                response = await client.post(full_url, json=json_data)
            elif method.upper() == "GET":
                response = await client.get(full_url)
            elif method.upper() == "DELETE":
                response = await client.delete(full_url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            return response
    
    async def proxy_streaming_request(
        self,
        backend: BackendType,
        endpoint: str,
        json_data: Dict,
        timeout: float = 300.0
    ):
        """
        Proxy a streaming request to a backend engine service
        
        Args:
            backend: Backend type to route to
            endpoint: API endpoint (e.g., "/v1/chat/completions")
            json_data: JSON request body
            timeout: Request timeout in seconds
            
        Yields:
            Response chunks as they arrive
        """
        url = self.get_engine_url(backend)
        full_url = f"{url}{endpoint}"
        
        logger.info(f"Proxying streaming request to {full_url}")
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", full_url, json=json_data) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk


# Global engine router instance
engine_router = EngineRouter()

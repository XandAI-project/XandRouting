import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


def add_cors_middleware(app: FastAPI):
    """
    Add CORS middleware to FastAPI application
    
    Configures CORS settings based on environment variables:
    - CORS_ORIGINS: Comma-separated list of allowed origins (default: *)
    - CORS_CREDENTIALS: Allow credentials (default: true)
    - CORS_METHODS: Allowed methods (default: *)
    - CORS_HEADERS: Allowed headers (default: *)
    
    Args:
        app: FastAPI application instance
    """
    # Get CORS configuration from environment
    cors_origins = os.getenv("CORS_ORIGINS", "*")
    cors_credentials = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"
    cors_methods = os.getenv("CORS_METHODS", "*")
    cors_headers = os.getenv("CORS_HEADERS", "*")
    
    # Parse origins (support comma-separated list)
    if cors_origins == "*":
        origins = ["*"]
    else:
        origins = [origin.strip() for origin in cors_origins.split(",")]
    
    # Parse methods
    if cors_methods == "*":
        methods = ["*"]
    else:
        methods = [method.strip() for method in cors_methods.split(",")]
    
    # Parse headers
    if cors_headers == "*":
        headers = ["*"]
    else:
        headers = [header.strip() for header in cors_headers.split(",")]
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=cors_credentials,
        allow_methods=methods,
        allow_headers=headers,
        expose_headers=["*"]
    )
    
    logger.info("CORS middleware configured:")
    logger.info(f"  Origins: {origins}")
    logger.info(f"  Credentials: {cors_credentials}")
    logger.info(f"  Methods: {methods}")
    logger.info(f"  Headers: {headers}")

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class DownloadStatus(str, Enum):
    """Download job status"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DownloadRequest(BaseModel):
    """Request to download a model from HuggingFace"""
    url: str = Field(..., description="Full HuggingFace URL (e.g., https://huggingface.co/org/model)")
    destination: Optional[str] = Field(None, description="Target directory name in /models/, defaults to model name")
    include: Optional[List[str]] = Field(None, description="File patterns to include (e.g., ['*.gguf', '*.safetensors'])")
    exclude: Optional[List[str]] = Field(None, description="File patterns to exclude (e.g., ['*.bin'])")
    quantization: Optional[str] = Field(None, description="GGUF quantization type (e.g., 'IQ4_XS', 'Q4_K_M')")
    
    @validator('url')
    def validate_url(cls, v):
        """Ensure URL is a valid HuggingFace URL"""
        if not v.startswith('http'):
            raise ValueError("URL must start with http:// or https://")
        if 'huggingface.co' not in v:
            raise ValueError("URL must be a HuggingFace URL")
        return v
    
    @validator('quantization')
    def validate_quantization(cls, v, values):
        """Warn if quantization is set without GGUF focus"""
        if v and values.get('include') and '*.gguf' not in values['include']:
            # Just a warning, still allow it
            pass
        return v


class DownloadJob(BaseModel):
    """Download job with status tracking"""
    job_id: str = Field(..., description="Unique job identifier")
    status: DownloadStatus = Field(..., description="Current job status")
    url: str = Field(..., description="Source HuggingFace URL")
    destination: str = Field(..., description="Destination path in /models/")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Download start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Download completion timestamp")
    duration_seconds: Optional[int] = Field(None, description="Total download duration")
    files_downloaded: Optional[int] = Field(None, description="Number of files downloaded")
    message: Optional[str] = Field(None, description="Status message")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DownloadListResponse(BaseModel):
    """Response for listing all download jobs"""
    downloads: List[DownloadJob]
    total: int


class DownloadJobSummary(BaseModel):
    """Summary information for a download job"""
    job_id: str
    status: DownloadStatus
    destination: str
    created_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

import logging
import threading
import time
import os
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path

from download_models import DownloadRequest, DownloadJob, DownloadStatus, DownloadJobSummary

logger = logging.getLogger(__name__)


class DownloadManager:
    """
    Manages background model downloads from HuggingFace
    Thread-safe job tracking and status updates
    """
    
    def __init__(self):
        """Initialize download manager"""
        self.jobs: Dict[str, DownloadJob] = {}
        self.lock = threading.RLock()
        self.active_workers: Dict[str, threading.Thread] = {}
        
        logger.info("DownloadManager initialized")
    
    def create_job(self, request: DownloadRequest) -> str:
        """
        Create new download job and start background worker
        
        Args:
            request: DownloadRequest with URL and options
            
        Returns:
            job_id for status polling
        """
        job_id = f"download_{int(time.time() * 1000)}"
        
        # Parse destination
        destination = self._parse_destination(request)
        
        # Create job
        job = DownloadJob(
            job_id=job_id,
            status=DownloadStatus.PENDING,
            url=request.url,
            destination=f"/models/{destination}",
            created_at=datetime.now(),
            message="Download job created"
        )
        
        with self.lock:
            self.jobs[job_id] = job
        
        logger.info(f"Created download job: {job_id} for {request.url}")
        
        # Start background worker thread
        worker = threading.Thread(
            target=self._download_worker,
            args=(job_id, request),
            daemon=True,
            name=f"DownloadWorker-{job_id}"
        )
        worker.start()
        
        with self.lock:
            self.active_workers[job_id] = worker
        
        return job_id
    
    def _download_worker(self, job_id: str, request: DownloadRequest):
        """
        Background worker that performs the download
        
        Args:
            job_id: Job identifier
            request: Download request with parameters
        """
        logger.info(f"Starting download worker for job: {job_id}")
        
        try:
            # Update status to downloading
            self._update_job_status(job_id, DownloadStatus.DOWNLOADING, "Downloading model files...")
            self._update_job_field(job_id, 'started_at', datetime.now())
            
            # Import huggingface_hub
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise RuntimeError("huggingface-hub is not installed")
            
            # Parse repository ID from URL
            repo_id = self._parse_repo_id(request.url)
            logger.info(f"Parsed repo ID: {repo_id}")
            
            # Get destination
            job = self.jobs[job_id]
            destination = job.destination
            
            # Ensure destination directory exists
            os.makedirs(destination, exist_ok=True)
            
            # Build filter patterns
            allow_patterns = request.include
            ignore_patterns = request.exclude
            
            # Apply quantization filter (overrides include)
            if request.quantization:
                allow_patterns = [f"*{request.quantization}*.gguf"]
                logger.info(f"Filtering for quantization: {request.quantization}")
            
            logger.info(f"Downloading {repo_id} to {destination}")
            if allow_patterns:
                logger.info(f"Include patterns: {allow_patterns}")
            if ignore_patterns:
                logger.info(f"Exclude patterns: {ignore_patterns}")
            
            # Download from HuggingFace
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=destination,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    token=None,  # Public models only
                    resume_download=True,
                    revision="main"  # Explicitly specify the branch
                )
            except Exception as download_error:
                logger.error(f"Snapshot download failed: {type(download_error).__name__}: {download_error}")
                
                # If main branch fails, try without specifying revision
                logger.info("Retrying without specifying revision...")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=destination,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    token=None,
                    resume_download=True
                )
            
            # Count downloaded files
            files_downloaded = self._count_files(destination)
            
            # Calculate duration
            job = self.jobs[job_id]
            duration = int((datetime.now() - job.started_at).total_seconds())
            
            # Mark as completed
            with self.lock:
                job.status = DownloadStatus.COMPLETED
                job.completed_at = datetime.now()
                job.duration_seconds = duration
                job.files_downloaded = files_downloaded
                job.message = "Download completed successfully"
            
            logger.info(f"Download completed: {job_id} ({files_downloaded} files, {duration}s)")
            
        except Exception as e:
            logger.error(f"Download failed for job {job_id}: {e}", exc_info=True)
            
            # Mark as failed
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id].status = DownloadStatus.FAILED
                    self.jobs[job_id].error = str(e)
                    self.jobs[job_id].message = f"Download failed: {str(e)}"
        
        finally:
            # Clean up worker reference
            with self.lock:
                if job_id in self.active_workers:
                    del self.active_workers[job_id]
    
    def get_job(self, job_id: str) -> Optional[DownloadJob]:
        """
        Get job by ID
        
        Args:
            job_id: Job identifier
            
        Returns:
            DownloadJob or None if not found
        """
        with self.lock:
            return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[DownloadJob]:
        """
        List all download jobs
        
        Returns:
            List of all jobs
        """
        with self.lock:
            return list(self.jobs.values())
    
    def list_job_summaries(self) -> List[DownloadJobSummary]:
        """
        List job summaries (lighter weight)
        
        Returns:
            List of job summaries
        """
        with self.lock:
            return [
                DownloadJobSummary(
                    job_id=job.job_id,
                    status=job.status,
                    destination=job.destination,
                    created_at=job.created_at
                )
                for job in self.jobs.values()
            ]
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a download job (if still running)
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancelled, False if not found or already completed
        """
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            
            # Can only cancel pending or downloading jobs
            if job.status not in [DownloadStatus.PENDING, DownloadStatus.DOWNLOADING]:
                return False
            
            # Mark as cancelled
            job.status = DownloadStatus.CANCELLED
            job.message = "Download cancelled by user"
            
            logger.info(f"Cancelled download job: {job_id}")
            
            # Note: The worker thread will check status and exit
            # We don't forcibly kill threads
            
            return True
    
    def _update_job_status(self, job_id: str, status: DownloadStatus, message: Optional[str] = None):
        """
        Update job status
        
        Args:
            job_id: Job identifier
            status: New status
            message: Optional status message
        """
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].status = status
                if message:
                    self.jobs[job_id].message = message
    
    def _update_job_field(self, job_id: str, field: str, value):
        """
        Update a specific job field
        
        Args:
            job_id: Job identifier
            field: Field name
            value: New value
        """
        with self.lock:
            if job_id in self.jobs:
                setattr(self.jobs[job_id], field, value)
    
    def _parse_repo_id(self, url: str) -> str:
        """
        Parse HuggingFace repository ID from URL
        
        Examples:
            https://huggingface.co/Qwen/Qwen3-Coder -> Qwen/Qwen3-Coder
            https://huggingface.co/meta-llama/Llama-2-7b -> meta-llama/Llama-2-7b
            https://huggingface.co/org/model/tree/main -> org/model
        
        Args:
            url: Full HuggingFace URL
            
        Returns:
            Repository ID in format org/model
            
        Raises:
            ValueError: If URL is invalid
        """
        # Clean up URL
        url = url.rstrip('/')
        
        # Remove common suffixes
        url = url.replace('/tree/main', '')
        url = url.replace('/tree/master', '')
        
        # Extract org/model part
        if 'huggingface.co/' in url:
            parts = url.split('huggingface.co/')
            if len(parts) > 1:
                repo_path = parts[1].split('?')[0]  # Remove query params
                return repo_path
        
        raise ValueError(f"Invalid HuggingFace URL: {url}")
    
    def _parse_destination(self, request: DownloadRequest) -> str:
        """
        Generate destination directory name
        
        Args:
            request: Download request
            
        Returns:
            Destination directory name
        """
        if request.destination:
            return request.destination
        
        # Use last part of repo ID
        try:
            repo_id = self._parse_repo_id(request.url)
            model_name = repo_id.split('/')[-1]
            
            # Clean up name
            model_name = model_name.lower()
            model_name = model_name.replace(' ', '-')
            
            return model_name
        except Exception as e:
            logger.warning(f"Could not parse destination from URL: {e}")
            # Fallback to timestamp-based name
            return f"model_{int(time.time())}"
    
    def _count_files(self, directory: str) -> int:
        """
        Count files in directory recursively
        
        Args:
            directory: Directory path
            
        Returns:
            Number of files
        """
        try:
            count = 0
            for root, dirs, files in os.walk(directory):
                count += len(files)
            return count
        except Exception as e:
            logger.warning(f"Could not count files in {directory}: {e}")
            return 0
    
    def get_active_downloads(self) -> List[str]:
        """
        Get list of currently active download job IDs
        
        Returns:
            List of job IDs
        """
        with self.lock:
            return [
                job_id for job_id, job in self.jobs.items()
                if job.status in [DownloadStatus.PENDING, DownloadStatus.DOWNLOADING]
            ]
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """
        Clean up completed jobs older than max_age_hours
        
        Args:
            max_age_hours: Maximum age in hours
        """
        with self.lock:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            
            jobs_to_remove = []
            for job_id, job in self.jobs.items():
                if job.status in [DownloadStatus.COMPLETED, DownloadStatus.FAILED, DownloadStatus.CANCELLED]:
                    if job.created_at.timestamp() < cutoff_time:
                        jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
            
            if jobs_to_remove:
                logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")


# Global download manager instance
download_manager = DownloadManager()

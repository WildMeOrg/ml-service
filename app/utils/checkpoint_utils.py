import os
import logging
import requests
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def is_url(path: str) -> bool:
    """Check if a path is a URL."""
    return path.startswith(('http://', 'https://'))

def download_checkpoint(url: str, cache_dir: str = "/tmp/checkpoints") -> str:
    """
    Download a checkpoint from a URL and cache it locally.
    
    Args:
        url: URL to download the checkpoint from
        cache_dir: Directory to cache downloaded checkpoints
        
    Returns:
        Local path to the downloaded checkpoint
        
    Raises:
        Exception: If download fails
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate filename from URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = "checkpoint.pth"
    
    local_path = os.path.join(cache_dir, filename)
    
    # Check if file already exists
    if os.path.exists(local_path):
        logger.info(f"Checkpoint already cached at {local_path}")
        return local_path
    
    logger.info(f"Downloading checkpoint from {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Checkpoint downloaded and cached at {local_path}")
        return local_path
        
    except Exception as e:
        logger.error(f"Failed to download checkpoint from {url}: {str(e)}")
        # Clean up partial file if it exists
        if os.path.exists(local_path):
            os.remove(local_path)
        raise

def get_checkpoint_path(checkpoint_path: Optional[str]) -> Optional[str]:
    """
    Get the local path to a checkpoint, downloading if necessary.
    
    Args:
        checkpoint_path: URL or local path to checkpoint, or None
        
    Returns:
        Local path to checkpoint or None if no checkpoint specified
    """
    if not checkpoint_path:
        return None
    
    if is_url(checkpoint_path):
        return download_checkpoint(checkpoint_path)
    else:
        # Verify local path exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        return checkpoint_path

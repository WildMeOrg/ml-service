import contextlib
import fcntl
import hashlib
import os
import logging
import queue
import tempfile
import threading
import requests
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# (connect, read) seconds. The read timeout bounds per-read inactivity only;
# DOWNLOAD_TOTAL_DEADLINE bounds wall-clock time for the whole fetch (headers
# included) so a drip-feeding server cannot block startup indefinitely.
DOWNLOAD_TIMEOUT = (10, 60)
DOWNLOAD_TOTAL_DEADLINE = 900

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

    # Key the cache by the full URL, not just the basename, so two stores
    # serving files with the same name (e.g. .../model.pt) don't collide.
    parsed_url = urlparse(url)
    basename = os.path.basename(parsed_url.path) or "checkpoint.pth"
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    local_path = os.path.join(cache_dir, f"{url_hash}_{basename}")

    if os.path.exists(local_path):
        logger.info(f"Checkpoint already cached at {local_path}")
        return local_path

    # Serialize downloaders across processes (multiple uvicorn workers or
    # replicas sharing a cache volume) so only one fetches; the rest wait on
    # the lock and reuse its result.
    with open(local_path + ".lock", 'w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.exists(local_path):
                logger.info(f"Checkpoint already cached at {local_path}")
                return local_path
            return _fetch_to_cache(url, cache_dir, local_path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

def _fetch_to_cache(url: str, cache_dir: str, local_path: str) -> str:
    """Fetch url to local_path, bounding total wall-clock time.

    The socket read timeout only limits per-read inactivity — a server that
    keeps trickling bytes can hold a blocking read far past any deadline
    checked between reads. So the fetch runs in a worker thread and the
    caller waits at most DOWNLOAD_TOTAL_DEADLINE, closing the response to
    abort the worker on expiry. An aborted worker may linger until its
    socket timeout fires, but the caller always regains control on time.
    """
    result_queue = queue.Queue(maxsize=1)
    response_holder = {}
    cancel = threading.Event()
    # Serializes cancellation against the worker's final check-and-rename so
    # a checkpoint can never be installed after cancel is set: either the
    # commit wins (install completed before the deadline path set cancel) or
    # cancellation wins (no install, ever).
    commit_lock = threading.Lock()

    def _worker():
        try:
            result_queue.put(
                ("ok", _stream_to_cache(url, cache_dir, local_path,
                                        response_holder, cancel, commit_lock))
            )
        except BaseException as e:
            result_queue.put(("err", e))

    thread = threading.Thread(target=_worker, daemon=True, name="checkpoint-download")
    thread.start()
    try:
        status, value = result_queue.get(timeout=DOWNLOAD_TOTAL_DEADLINE)
    except queue.Empty:
        # Order matters: set cancel before reading the holder. Either we see
        # the response and close it, or the worker publishes afterwards and
        # then sees cancel — no interleaving lets it run to completion.
        with commit_lock:
            cancel.set()
        response = response_holder.get("response")
        if response is not None:
            with contextlib.suppress(Exception):
                response.close()
        logger.error(
            f"Download of {url} exceeded {DOWNLOAD_TOTAL_DEADLINE}s deadline"
        )
        raise TimeoutError(
            f"Download of {url} exceeded {DOWNLOAD_TOTAL_DEADLINE}s deadline"
        )
    if status == "err":
        raise value
    return value

def _stream_to_cache(url: str, cache_dir: str, local_path: str,
                     response_holder: dict, cancel: threading.Event,
                     commit_lock: threading.Lock) -> str:
    """Stream url to a temp file, then rename into place so a concurrent
    reader (or a crash mid-download) never sees a partial file at local_path.

    Checks `cancel` after publishing the response, while streaming, and
    before the final rename, so a caller that has already timed out can
    guarantee no checkpoint is installed after it raised.
    """
    logger.info(f"Downloading checkpoint from {url}")
    tmp_path = None
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
            response_holder["response"] = response
            if cancel.is_set():
                raise TimeoutError(f"Download of {url} cancelled after deadline")
            response.raise_for_status()

            fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".part")
            try:
                f = os.fdopen(fd, 'wb')
            except Exception:
                os.close(fd)
                raise
            with f:
                for chunk in response.iter_content(chunk_size=8192):
                    if cancel.is_set():
                        raise TimeoutError(
                            f"Download of {url} cancelled after deadline"
                        )
                    f.write(chunk)
        with commit_lock:
            if cancel.is_set():
                raise TimeoutError(f"Download of {url} cancelled after deadline")
            os.replace(tmp_path, local_path)

        logger.info(f"Checkpoint downloaded and cached at {local_path}")
        return local_path

    except Exception as e:
        logger.error(f"Failed to download checkpoint from {url}: {str(e)}")
        if tmp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp_path)
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

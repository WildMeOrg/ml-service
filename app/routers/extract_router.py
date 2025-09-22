import logging
from fastapi import APIRouter, HTTPException, Request, status, Depends
from typing import Dict, Any, List, Optional
import httpx
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from app.models.model_handler import ModelHandler
from app.models.miewid import MiewidModel
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extract", tags=["Embeddings Extraction"])

# Limit concurrent extractions to prevent OOM errors
MAX_CONCURRENT_EXTRACTIONS = 2
extract_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTIONS)

def is_url(string: str) -> bool:
    """Check if a string is a valid URL.
    
    Args:
        string: The string to check
        
    Returns:
        bool: True if the string is a URL, False otherwise
    """
    return string.startswith(('http://', 'https://'))

class ExtractRequest(BaseModel):
    """Request model for embeddings extraction endpoint."""
    model_id: str = Field(..., description="ID of the MiewID model to use for extraction")
    image_uri: str = Field(..., description="URI of the image to process (URL or file path)")
    bbox: List[int] = Field(None, description="Optional bounding box coordinates [x, y, width, height]. If not provided, uses full image")
    theta: float = Field(default=0.0, description="Rotation angle in radians")

async def get_model_handler(request: Request) -> ModelHandler:
    """Dependency to get the model handler from the app state."""
    return request.app.state.model_handler

@router.post("/", response_model=Dict[str, Any])
async def extract_embeddings(
    extract_request: ExtractRequest,
    handler: ModelHandler = Depends(get_model_handler)
):
    """Extract embeddings from an image using MiewID model with bounding box and rotation.
    
    Args:
        extract_request: The extraction request containing model_id, image_uri, bbox, and theta
        handler: The model handler instance
        
    Returns:
        Dictionary containing the embeddings and metadata
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    async with extract_semaphore:
        try:
            # Validate bbox format if provided
            if extract_request.bbox is not None and len(extract_request.bbox) != 4:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Bounding box must contain exactly 4 values: [x, y, width, height]"
                )
            
            # Get the model instance
            model = handler.get_model(extract_request.model_id)
            if not model:
                available_models = list(handler.list_models().keys())
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": f"Model '{extract_request.model_id}' not found.",
                        "available_models": available_models
                    }
                )
            
            # Check if the model is a MiewID model
            if not isinstance(model, MiewidModel):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{extract_request.model_id}' is not a MiewID model. Only MiewID models support embeddings extraction."
                )
            
            # Download image if it's a URL
            if is_url(extract_request.image_uri):
                async with httpx.AsyncClient() as client:
                    response = await client.get(extract_request.image_uri)
                    response.raise_for_status()
                    image_bytes = response.content
            else:
                # Handle local file path
                file_path = Path(extract_request.image_uri)
                if not file_path.exists():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File not found: {extract_request.image_uri}"
                    )
                with open(file_path, "rb") as f:
                    image_bytes = f.read()
            
            # Convert bbox to tuple if provided
            bbox_tuple = tuple(extract_request.bbox) if extract_request.bbox is not None else None
            
            # Extract embeddings in a thread pool
            embeddings = await run_in_threadpool(
                model.extract_embeddings,
                image_bytes=image_bytes,
                bbox=bbox_tuple,
                theta=extract_request.theta
            )
            
            # Prepare response
            result = {
                'model_id': extract_request.model_id,
                'embeddings': embeddings.tolist(),  # Convert numpy array to list for JSON serialization
                'embeddings_shape': list(embeddings.shape),
                'bbox': extract_request.bbox,
                'theta': extract_request.theta,
                'image_uri': extract_request.image_uri
            }
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Error downloading image: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error downloading image: {str(e)}"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Embeddings extraction error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Embeddings extraction error: {str(e)}"
            )

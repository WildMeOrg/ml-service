import logging
from fastapi import APIRouter, HTTPException, Request, status, Depends
from typing import Dict, Any, List, Optional
import httpx
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from app.models.model_handler import ModelHandler
from app.models.efficientnet import EfficientNetModel
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/classify", tags=["Image Classification"])

# Limit concurrent classifications to prevent OOM errors
MAX_CONCURRENT_CLASSIFICATIONS = 2
classify_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLASSIFICATIONS)

def is_url(string: str) -> bool:
    """Check if a string is a valid URL.
    
    Args:
        string: The string to check
        
    Returns:
        bool: True if the string is a URL, False otherwise
    """
    return string.startswith(('http://', 'https://'))

class ClassifyRequest(BaseModel):
    """Request model for image classification endpoint."""
    model_id: str = Field(..., description="ID of the EfficientNet model to use for classification")
    image_uri: str = Field(..., description="URI of the image to process (URL or file path)")
    bbox: List[int] = Field(None, description="Optional bounding box coordinates [x, y, width, height]. If not provided, uses full image")
    theta: float = Field(default=0.0, description="Rotation angle in radians")

async def get_model_handler(request: Request) -> ModelHandler:
    """Dependency to get the model handler from the app state."""
    return request.app.state.model_handler

@router.post("/", response_model=Dict[str, Any])
async def classify_image(
    classify_request: ClassifyRequest,
    handler: ModelHandler = Depends(get_model_handler)
):
    """Classify an image using EfficientNet model with bounding box and rotation.
    
    Args:
        classify_request: The classification request containing model_id, image_uri, bbox, and theta
        handler: The model handler instance
        
    Returns:
        Dictionary containing the classification results
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    async with classify_semaphore:
        try:
            # Validate bbox format if provided
            if classify_request.bbox is not None and len(classify_request.bbox) != 4:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Bounding box must contain exactly 4 values: [x, y, width, height]"
                )
            
            # Get the model instance
            model = handler.get_model(classify_request.model_id)
            if not model:
                available_models = list(handler.list_models().keys())
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": f"Model '{classify_request.model_id}' not found.",
                        "available_models": available_models
                    }
                )
            
            # Check if the model is an EfficientNet model
            if not isinstance(model, EfficientNetModel):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{classify_request.model_id}' is not an EfficientNet model. Only EfficientNet models support classification."
                )
            
            # Download image if it's a URL
            if is_url(classify_request.image_uri):
                async with httpx.AsyncClient() as client:
                    response = await client.get(classify_request.image_uri)
                    response.raise_for_status()
                    image_bytes = response.content
            else:
                # Handle local file path
                file_path = Path(classify_request.image_uri)
                if not file_path.exists():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File not found: {classify_request.image_uri}"
                    )
                with open(file_path, "rb") as f:
                    image_bytes = f.read()
            
            # Run classification in a thread pool
            result = await run_in_threadpool(
                model.predict,
                image_bytes=image_bytes,
                bbox=classify_request.bbox,
                theta=classify_request.theta
            )
            
            # Add request metadata to result
            result['image_uri'] = classify_request.image_uri
            
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
            logger.error(f"Image classification error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Image classification error: {str(e)}"
            )

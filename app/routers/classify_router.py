import logging
from fastapi import APIRouter, HTTPException, Request, status, Depends
from typing import Dict, Any, List, Optional
import httpx
import asyncio
from pydantic import BaseModel, Field
from app.models.model_handler import ModelHandler
from app.utils.image_uri import resolve_image_uri, sanitize_uri_for_response
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/classify", tags=["Image Classification"])

# Limit concurrent classifications to prevent OOM errors
MAX_CONCURRENT_CLASSIFICATIONS = 2
classify_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLASSIFICATIONS)

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
            
            # Check if the model supports classification (has predict method)
            if not hasattr(model, 'predict'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{classify_request.model_id}' does not support classification."
                )

            # Resolve image bytes from URI (URL, data URI, or local path)
            try:
                image_bytes = await resolve_image_uri(classify_request.image_uri)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
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
            result['image_uri'] = sanitize_uri_for_response(classify_request.image_uri)
            
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

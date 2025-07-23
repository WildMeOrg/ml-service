import logging
from fastapi import APIRouter, HTTPException, Request, status, Depends
from typing import Optional, Dict, Any, List
import httpx
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
import json
import os
from app.models.model_handler import ModelHandler
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Limit concurrent predictions to prevent OOM errors
MAX_CONCURRENT_PREDICTIONS = 2
predict_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PREDICTIONS)

def is_url(string: str) -> bool:
    """Check if a string is a valid URL.
    
    Args:
        string: The string to check
        
    Returns:
        bool: True if the string is a URL, False otherwise
    """
    return string.startswith(('http://', 'https://'))

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    model_id: str = Field(..., description="ID of the model to use for prediction")
    image_uri: str = Field(..., description="URI of the image to process (URL or file path)")
    model_params: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional parameters to override model configuration"
    )

class ModelInfo(BaseModel):
    """Response model for model information endpoint."""
    model_id: str
    handler_type: str
    parameters: Dict[str, Any]

async def get_model_handler(request: Request) -> ModelHandler:
    """Dependency to get the model handler from the app state."""
    return request.app.state.model_handler

@router.get("/models", response_model=Dict[str, Any])
async def list_models(handler: ModelHandler = Depends(get_model_handler)):
    """List all available models and their information.
    
    Returns:
        Dict containing information about all loaded models
    """
    try:
        return handler.list_models()
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )

@router.post("/", response_model=Dict[str, Any])
async def predict(
    prediction: PredictionRequest,
    handler: ModelHandler = Depends(get_model_handler)
):
    """Run prediction on an image using the specified model.
    
    Args:
        prediction: The prediction request containing model_id and image_uri
        handler: The model handler instance
        
    Returns:
        The prediction results
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    async with predict_semaphore:
        try:
            # Get the model instance
            model = handler.get_model(prediction.model_id)
            if not model:
                available_models = list(handler.list_models().keys())
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": f"Model '{prediction.model_id}' not found.",
                        "available_models": available_models
                    }
                )
            
            # Download image if it's a URL
            if is_url(prediction.image_uri):
                async with httpx.AsyncClient() as client:
                    response = await client.get(prediction.image_uri)
                    response.raise_for_status()
                    image_bytes = response.content
            else:
                # Handle local file path
                file_path = Path(prediction.image_uri)
                if not file_path.exists():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File not found: {prediction.image_uri}"
                    )
                with open(file_path, "rb") as f:
                    image_bytes = f.read()
            
            # Get model info for default parameters
            model_info = handler.get_model_info(prediction.model_id)
            
            # Prepare prediction parameters, overriding defaults with any provided parameters
            predict_params = model_info['config'].copy()
            if prediction.model_params:
                predict_params.update(prediction.model_params)
            
            # Remove any parameters that shouldn't be passed to predict
            predict_params.pop('model_path', None)
            predict_params.pop('device', None)
            
            # Run inference in a thread pool
            result = await run_in_threadpool(
                model.predict,
                image_bytes=image_bytes,
                **predict_params
            )
            
            # Add model_id to the result
            result['model_id'] = prediction.model_id
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
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction error: {str(e)}"
            )
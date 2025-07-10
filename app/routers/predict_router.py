import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import Optional
import httpx
import asyncio
from pathlib import Path
from app.schemas.model_response import ModelResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])

N = 1
predict_semaphore = asyncio.Semaphore(N)

from pydantic import BaseModel

def is_url(string):
    return string.startswith(('http://', 'https://'))

class PredictionRequest(BaseModel):
    model_id: str
    image_uri: str

@router.post("/")
async def predict(
    request: Request,
    prediction: PredictionRequest
):
    logger.info("Received prediction request for model_id: %s", prediction.model_id)
    
    image_uri = prediction.image_uri.strip()
    image_bytes = None
    
    try:
        if is_url(image_uri):
            async with httpx.AsyncClient() as client:
                response = await client.get(image_uri)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {response.status_code}")
            image_bytes = response.content
            logger.info(f"Successfully downloaded image from URL: {image_uri}")
        else:
            # Handle as file path
            image_path = str(Path(image_uri).expanduser().resolve())
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            logger.info(f"Successfully loaded image from path: {image_path}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from URI '{image_uri}': {str(e)}")
    
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image data could be loaded.")

    yolo_handler = request.app.state.yolo_handler
    model_id = prediction.model_id
    
    # Only apply semaphore to the actual prediction
    async with predict_semaphore:
        model_response = yolo_handler.predict(model_id, image_bytes)
    
    return model_response
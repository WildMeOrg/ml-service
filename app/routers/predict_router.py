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

class PredictionRequest(BaseModel):
    model_id: str
    image_url: Optional[str] = None
    image_path: Optional[str] = None

@router.post("/")
async def predict(
    request: Request,
    prediction: PredictionRequest
):
    logger.info("Received prediction request for model_id: %s", prediction.model_id)
    
    if not any([prediction.image_url, prediction.image_path]):
        raise HTTPException(status_code=400, detail="At least one image source (URL or path) must be provided")
    
    image_bytes = None
    
    if prediction.image_url:
        async with httpx.AsyncClient() as client:
            response = await client.get(prediction.image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image from URL.")
        image_bytes = response.content
    
    elif prediction.image_path:
        try:
            image_path = str(Path(prediction.image_path).resolve())
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load image from path: {str(e)}")
    
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image data could be loaded.")

    yolo_handler = request.app.state.yolo_handler
    model_id = prediction.model_id
    
    # Only apply semaphore to the actual prediction
    async with predict_semaphore:
        results = yolo_handler.predict(model_id, image_bytes)
    
    results = results[0]

    names = yolo_handler.models[model_id]["model"].names

    bboxes = results.boxes.xywh.tolist()
    scores = results.boxes.conf.tolist()
    class_ids = results.boxes.cls.tolist()
    class_names = [names[class_id] for class_id in class_ids]

    thetas = [0]*len(bboxes)

    return ModelResponse(
        bboxes=bboxes,
        scores=scores,
        thetas=thetas,
        class_ids=class_ids,
        class_names=class_names
    )
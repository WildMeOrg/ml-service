import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import Optional
import httpx
import asyncio
from pathlib import Path
from app.schemas.model_response import ModelResponse
import torch
from torch.utils.data import DataLoader
import cv2
from pydantic import BaseModel


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["Explain"])

N = 1
explain_semaphore = asyncio.Semaphore(N)

def is_url(string):
    return string.startswith(('http://', 'https://'))

class body(BaseModel):
    path1: str
    name1: str
    bb1: list[int]
    theta1: float
    path1: str
    name1: str
    bb2: list[int]
    theta2: float
    model_id: str
    crop_bbox: bool = False
    visualization_type: str = "lines_and_colors"
    layer_key: str = "backbone.blocks.3"
    k_lines: int = 20
    k_colors: int = 10

@router.post("/")
async def read_items(
    request: Request,
    body: body
    ):

    image1_uri = body.image1_uri.strip()
    image1_bytes = None
    
    try:
        if is_url(image1_uri):
            async with httpx.AsyncClient() as client:
                response = await client.get(image1_uri)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {response.status_code}")
            image1_bytes = response.content
            logger.info(f"Successfully downloaded image from URL: {image1_uri}")
        else:
            # Handle as file path
            image1_path = str(Path(image1_uri).expanduser().resolve())
            with open(image1_path, 'rb') as f:
                image1_bytes = f.read()
            logger.info(f"Successfully loaded image from path: {image1_path}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from URI '{image1_uri}': {str(e)}")
    
    image2_uri = body.image1_uri.strip()
    image2_bytes = None
    
    
    try:
        if is_url(image2_uri):
            async with httpx.AsyncClient() as client:
                response = await client.get(image2_uri)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {response.status_code}")
            image2_bytes = response.content
            logger.info(f"Successfully downloaded image from URL: {image2_uri}")
        else:
            # Handle as file path
            image2_path = str(Path(image2_uri).expanduser().resolve())
            with open(image2_path, 'rb') as f:
                image2_bytes = f.read()
            logger.info(f"Successfully loaded image from path: {image2_path}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from URI '{image2_uri}': {str(e)}")

    # Verify parameters:
    if len(body.bb1) != 4:
        raise HTTPException(status_code=400, detail=f"Bad bounding box 1")
    if len(body.bb2) != 4:
        raise HTTPException(status_code=400, detail=f"Bad bounding box 2")
    for x in body.bb1:
        if x < 0:
            raise HTTPException(status_code=400, detail=f"Bad bounding box 1")
    for x in body.bb2:
        if x < 0:
            raise HTTPException(status_code=400, detail=f"Bad bounding box 2")
    if body.k_lines < 0:
        raise HTTPException(status_code=400, detail=f"K Lines must be positive")
    if body.k_lines > 99:
        raise HTTPException(status_code=400, detail=f"K Lines must be less than 100")
    if body.k_colors < 0:
        raise HTTPException(status_code=400, detail=f"K Colors must be positive")
    if body.k_colors > 99:
        raise HTTPException(status_code=400, detail=f"K Colors must be less than 100")



    yolo_handler = request.app.state.yolo_handler

    data = [[image1_bytes, "a", image1_path, body.bb1, body.theta1], [image2_bytes, "a", image2_path, body.bb2, body.theta2]]
    test_loader = DataLoader(data, batch_size=1, shuffle=False)
    
    # Only apply semaphore to the actual prediction
    async with explain_semaphore:
        model_response = yolo_handler.explain(test_loader, body.model_id, body.crop_bbox, body.visualization_type, body.layer_key, body.k_lines, body.k_colors)

    cv2.imwrite("response.png", model_response)
    return model_response
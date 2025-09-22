import asyncio
import logging
from pathlib import Path

import cv2
import httpx
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from pairx import explain

from app.models.model_handler import ModelHandler
from app.utils.helpers import get_chip_from_img

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["Explain"])

MAX_BATCH_SIZE = 16
MAX_CONCURRENT_EXPLANATIONS = 2
explain_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXPLANATIONS)

def is_url(string):
    """Checks if a string is formatted as a url"""
    return string.startswith(('http://', 'https://'))

async def get_model_handler(request: Request) -> ModelHandler:
    """Dependency to get the model handler from the app state."""
    return request.app.state.model_handler

def preprocess(image, model):
    """Runs preprocessing on an image based on the model to be used."""
    image = Image.fromarray(image.astype("uint8"))
    if model.lower() == "miewid-msv3":
        transform = transforms.Compose([
            transforms.Resize((440, 440)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
         raise HTTPException(status_code=400, detail="Unsupported model")
    return transform(image)

def extend_bb_list(img_list, bb_list):
    """Extends a list a bounding boxes to the length of a list of images.
    Values added mean that no bounding takes place"""
    for x in range(len(img_list) - len(bb_list)):
        bb_list.append([0, 0, 0, 0])
    return bb_list

def extend_theta_list(img_list, theta_list):
    """Extends a list of thetas to the length of a list of images.
    Thetas added mean that no rotation takes place"""
    for x in range(len(img_list) - len(theta_list)):
        theta_list.append(0.0)
    return theta_list 

def validate_img_parameters(bbox, theta):
    """Ensure that a bounding box and theta are valid"""
    if len(bbox) != 4:
        raise HTTPException(status_code=400, detail=f"Each bounding box should have 4 values")
    for x in bbox:
        if x < 0:
            raise HTTPException(status_code=400, detail="Bounding box values should be positive")
    if theta < 0:
        raise HTTPException(status_code=400, detail="Theta should be greater than 0")

def validate_vis_parameters(body):
    """Checks if body parameters related to a specific visualization algorithm are valid."""
    if body.algorithm.lower() == "pairx":
        if body.k_lines < 0:
            raise HTTPException(status_code=400, detail=f"K Lines must be positive")
        if body.k_lines > 99:
            raise HTTPException(status_code=400, detail=f"K Lines must be less than 100")
        if body.k_colors < 0:
            raise HTTPException(status_code=400, detail=f"K Colors must be positive")
        if body.k_colors > 99:
            raise HTTPException(status_code=400, detail=f"K Colors must be less than 100")
        if body.visualization_type not in ["lines_and_colors", "only_lines", "only_colors"]:
            raise HTTPException(status_code=400, detail="Unsupported visualization type.")
        possible_models = ["miewid-msv3"]
        if not body.model_id.lower() in possible_models:
            raise HTTPException(status_code=400, detail="Unsupported model for pairx.")
    else:
        raise HTTPException(status_code=400, detail="Unsupported algorithm.")

async def process_image(uri, bbox, theta, crop_bbox, model, device):
    """Reads image in from uri and generates pretransform and transform images to use for visualiztaion. 
    If crop_bbox is true, the preptransform image will be cropped. The transformed image will always be cropped.
    The transformed image will be stored on the device provided, ("cpu", "cuda", etc.)"""
    uri = uri.strip()
    try:
        if is_url(uri):
            async with httpx.AsyncClient() as client:
                response = await client.get(uri)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download image: {response.status_code}")
            image_bytes = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            path = str(Path(uri).expanduser().resolve())
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")

    validate_img_parameters(bbox, theta)

    chip = get_chip_from_img(image, bbox, theta)
    transformed_image = preprocess(chip, model)
    if len(transformed_image.shape) == 3:
            transformed_image = transformed_image.unsqueeze(0)
    if crop_bbox:
        image = chip
    img_size = tuple(transformed_image.shape[-2:])
    image = np.array(transforms.Resize(img_size)(Image.fromarray(image)))
    return image, transformed_image.to(device)

def process_asyncio_result(result):
    """Processes a result of process_image() when it is run via asyncio."""
    if isinstance(result, Exception):
        raise HTTPException(status_code=400, detail=f"{str(result)}")
    else:
        image, transform = result
        return image, transform

def run_pairx(imgs1_transformed, imgs2_transformed, imgs1, imgs2, model, layer_key, 
        k_lines, k_colors, visualization_type):
    """Run PAIR-X on provided images with given parameters.
        
        Args:
            imgs1_transformed: List of transformed images
            imgs2_transformed: List of transformed images. Length should match imgs1_transformed
            imgs1: Untransformed counterparts of imgs1_transformed
            imgs2: Untransformed counterparts of imgs2_transformed
            model: Actual model to be used rather than model id
            layer_key: layer within the model to use for feature matching and relevance propagation. 
                Earlier layer keys lead to visualizations that are focused on very specific points. 
                Later layer keys lead to visualizations that encompass broad swaths of the image.
                Layer keys in the middle tend to be preferred qualitatively.
            k_lines: The number of points on the two images to be matched and connected with lines
                in the visualization. High values of k lines often lead to clearly erroneous matches,
                but do not significantly impact performance.
            k_colors:
                The number of matches to backpropagate relevance on. Higher values of k_colors make 
                the algorithm much slower.
            visualization_type: One of "lines_and_colors", "only_colors", or "only_lines". 
                "lines_and_colors" yields the entire visualization
                "only_colors" crops out the half to only show the backpropagated relevances.
                "only_lines" crops out the bottom half to only show the feature matches.
            
        Returns:
            List of completed visualizations
    """
    
    # There is no reason to do backpropagation if we are not going to display it.
    if visualization_type == "only_lines":
        k_colors = 0

    if not layer_key in dict(model.named_modules()):
        raise HTTPException(status_code=400, detail="Invalid layer key")

    pairx_imgs = []
    try:
        pairx_imgs = explain(
            torch.cat(imgs1_transformed),
            torch.cat(imgs2_transformed),
            imgs1,
            imgs2,
            model,
            [layer_key],
            k_lines=k_lines,
            k_colors=k_colors,
        )
    # Handle out of memory errors by breaking into two batches and running again
    except Exception as e:
        if e.startsWith("torch.cuda.OutOfMemoryError:"):
            dim_size = imgs1_transformed.shape[0]
            first_half = run_pairx(imgs1_transformed[:midpoint], imgs2_transformed[:midpoint], imgs1[:midpoint], imgs2[:midpoint], model,
                    layer_key, k_lines, k_colors, visualization_type)
            second_half = run_pairx(imgs1_transformed[midpoint:], imgs2_transformed[midpoint:], imgs1[midpoint:], imgs2[midpoint:], model,
                    layer_key, k_lines, k_colors, visualization_type)
            return first_half + second_half
        else:
            raise HTTPException(status_code=500, detail=f"Internal Server Error")
    
    toReturn = []
    for pairx_img in pairx_imgs:
        pairx_height = pairx_img.shape[0] // 2

        if visualization_type == "only_lines":
            pairx_img = pairx_img[:pairx_height]
        elif visualization_type == "only_colors":
            pairx_img = pairx_img[pairx_height:]

        pairx_img = cv2.cvtColor(pairx_img, cv2.COLOR_BGR2RGB)
        toReturn.append(pairx_img)
    return toReturn

class body(BaseModel):
    # API input parameters
    image1_uris: list[str]
    bb1: list[list[float]]
    theta1: list[float] = [0.0]
    image2_uris: list[str]
    bb2: list[list[float]]
    theta2: list[float] = [0.0]
    model_id: str = "miewid-msv3"
    crop_bbox: bool = False
    visualization_type: str = "lines_and_colors"
    layer_key: str = "backbone.blocks.3"
    k_lines: int = 20
    k_colors: int = 5
    algorithm: str = "pairx"

@router.post("/")
async def read_items(
    request: Request,
    body: body, 
    handler: ModelHandler = Depends(get_model_handler)
    ):

    validate_vis_parameters(body)
    device = request.app.state.device

    image1s = []
    image2s = []
    image1s_transformed = []
    image2s_transformed = []
    
    # Fill in missing bbs and thetas with values that result in no cropping
    bb1s = extend_bb_list(body.image1_uris, body.bb1)
    bb2s = extend_bb_list(body.image2_uris, body.bb2)
    theta1s = extend_theta_list(body.image1_uris, body.theta1)
    theta2s = extend_theta_list(body.image2_uris, body.theta2)

    # Read in images asynchronously
    tasks = []
    for uri, bb, theta in zip(body.image1_uris, bb1s, theta1s):
        tasks.append(process_image(uri, bb, theta, body.crop_bbox, body.model_id, device))
    results1 = await asyncio.gather(*tasks, return_exceptions=False)
    
    tasks = []
    for uri, bb, theta in zip(body.image2_uris, bb2s, theta2s):
        tasks.append(process_image(uri, bb, theta, body.crop_bbox, body.model_id, device))
    results2 = await asyncio.gather(*tasks, return_exceptions=True)
    

    if len(body.image1_uris) == 1:
        image1, image1_transformed = process_asyncio_result(results1[0])
        for result in results2:
            image1s.append(image1)
            image1s_transformed.append(image1_transformed)
            image2, image2_transformed = process_asyncio_result(result)
            image2s.append(image2)
            image2s_transformed.append(image2_transformed)
    else:
        if len(body.image1_uris) != len(body.image2_uris):
            raise HTTPException(status_code=400, detail="Either provide only one image 1 or the same number of image1s and image2s.")
        else:
            if len(body.image1_uris) > MAX_BATCH_SIZE:
                raise HTTPException(status_code=400, detail=f"Batch exceeded max size of {str(MAX_BATCH_SIZE)}")
            for i in range(len(body.image1_uris)):
                image1, image1_transformed = process_asyncio_result(results1[i])
                image1s.append(image1)
                image1s_transformed.append(image1_transformed)
                image2, image2_transformed = process_asyncio_result(results2[i])
                image2s.append(image2)
                image2s_transformed.append(image2_transformed)

    visualiztions = []
    # Only apply semaphore to the actual prediction
    async with explain_semaphore:
        if body.algorithm.lower() == "pairx":
            model = handler.get_model(body.model_id).model
            visualizations = run_pairx(image1s_transformed, image2s_transformed, image1s, image2s, model, body.layer_key, body.k_lines, body.k_colors, body.visualization_type)
        else:
            raise HTTPException(status_code=400, detail="Unsupported algorithm.")
    
    for i in range(len(visualizations)):
        cv2.imwrite("response"+str(i)+".png", visualizations[i])
    return {'response': 'visualizations'}

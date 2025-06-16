from io import BytesIO
from PIL import Image
from app.schemas.model_response import ModelResponse  # Import your response schema

def run_inference(image_bytes: bytes, model, device: str, imgsz: int, conf: float) -> ModelResponse:
    """
    Run object detection inference on an image, with postprocessing to dilate bounding boxes.

    This function processes an image and runs inference using the provided YOLO model.
    It supports both oriented bounding boxes (OBB) and standard axis-aligned bounding boxes.
    After detection, each bounding box is "dilated" — its width and height are increased 
    by 15% while maintaining the same center coordinates.

    Args:
        image_bytes (bytes): The input image as bytes.
        model: The loaded YOLO model instance.
        device (str): The device to run inference on ('cpu', 'cuda', etc.).
        imgsz (int): The size to which the image will be resized before inference.
        conf (float): Confidence threshold for detections (0-1).

    Returns:
        ModelResponse: A Pydantic model containing the detection results including:
            - bboxes: List of dilated bounding boxes in [x_min, y_min, w, h] format
            - scores: List of confidence scores for each detection
            - thetas: List of rotation angles for each detection (0.0 for axis-aligned)
            - class_ids: List of class IDs for each detection
            - class_names: List of class names for each detection

    Note:
        - For OBB results, thetas contain the predicted rotation angles.
        - For standard bounding boxes, thetas are all 0.0.
        - Bounding box dilation increases size by 15% in both width and height.
    """
    img = Image.open(BytesIO(image_bytes))
    results = model.predict(img, save=False, imgsz=imgsz, conf=conf, device=device, verbose=False)
    results = results[0]
    names = model.names

    dilation_factor = 0.15  # 15% dilation

    if hasattr(results, 'obb') and results.obb is not None:
        xywhr = results.obb.xywhr.cpu().numpy()
        bboxes = []
        for x, y, w, h, r in xywhr:
            w *= (1 + dilation_factor)
            h *= (1 + dilation_factor)
            bboxes.append([x - w / 2, y - h / 2, w, h])
        thetas = [float(r) for x, y, w, h, r in xywhr]
        scores = results.obb.conf.tolist()
        class_ids = [int(cls) for cls in results.obb.cls.tolist()]
        class_names = [names[class_id] for class_id in class_ids]

    elif hasattr(results, 'boxes') and results.boxes is not None:
        xywh = results.boxes.xywh.cpu().numpy()
        bboxes = []
        for x, y, w, h in xywh:
            w *= (1 + dilation_factor)
            h *= (1 + dilation_factor)
            bboxes.append([x - w / 2, y - h / 2, w, h])
        thetas = [0.0] * len(bboxes)
        scores = results.boxes.conf.tolist()
        class_ids = [int(cls) for cls in results.boxes.cls.tolist()]
        class_names = [names[class_id] for class_id in class_ids]

    else:
        return ModelResponse([], [], [], [], [])

    return ModelResponse(
        bboxes=bboxes,
        scores=scores,
        thetas=thetas,
        class_ids=class_ids,
        class_names=class_names
    )

import torch
from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from .pairx.core import explain
from .pairx.xai_dataset import XAIDataset
from .helpers import get_chip_from_img, load_image


def draw_one(
    device,
    test_loader,
    model,
    crop_bbox,
    visualization_type="lines_and_colors",
    layer_key="backbone.blocks.3",
    k_lines=20,
    k_colors=10,
):
    """
    Generates a PAIR-X explanation for the provided images and model.

    Args:
        device (str or torch.device): Device to use (cuda or cpu).
        test_loader (DataLoader): Should contain two images, with 4 items for each (image, name, path, bbox as xywh).
        model (torch.nn.Module or equivalent): The deep metric learning model.
        visualization_type (str): The part of the PAIR-X visualization to return, selected from "lines_and_colors" (default), "only_lines", and "only_colors".
        layer_keys (str): The key of the intermediate layer to be used for explanation. Defaults to 'backbone.blocks.3'.
        k_lines (int, optional): The number of matches to visualize as lines. Defaults to 20.
        k_colors (int, optional): The number of matches to backpropagate to original image pixels. Defaults to 10.

    Returns:
        numpy.ndarray: PAIR-X visualization of type visualization_type.
    """
    assert test_loader.batch_size == 1, "test_loader should have a batch size of 1"
    assert len(test_loader) == 2, "test_loader should only contain two images"
    assert visualization_type in (
        "lines_and_colors",
        "only_lines",
        "only_colors",
    ), "unsupported visualization type"

    transformed_images = []
    pretransform_images = []

    # get transformed and untransformed images out of test_loader
    for batch in test_loader:
        (transformed_image,), _, (path,), (bbox,), (theta,) = batch[:5]

        if len(transformed_image.shape) == 3:
            transformed_image = transformed_image.unsqueeze(0)

        transformed_images.append(transformed_image.to(device))

        img_size = tuple(transformed_image.shape[-2:])
        pretransform_image = load_image(path)

        if crop_bbox:
            pretransform_image = get_chip_from_img(pretransform_image, bbox, theta)

        pretransform_image = np.array(transforms.Resize(img_size)(Image.fromarray(pretransform_image)))
        pretransform_images.append(pretransform_image)

    img_0, img_1 = transformed_images
    img_np_0, img_np_1 = pretransform_images

    # If only returning image with lines, skip generating color maps to save time
    if visualization_type == "only_lines":
        k_colors = 0

    # generate explanation image and return
    model.eval()
    model.device = device
    pairx_img = explain(
        img_0,
        img_1,
        img_np_0,
        img_np_1,
        model,
        [layer_key],
        k_lines=k_lines,
        k_colors=k_colors,
    )

    pairx_height = pairx_img.shape[0] // 2

    if visualization_type == "only_lines":
        return pairx_img[:pairx_height]
    elif visualization_type == "only_colors":
        return pairx_img[pairx_height:]

    pairx_img = cv2.cvtColor(pairx_img, cv2.COLOR_BGR2RGB)

    return pairx_img


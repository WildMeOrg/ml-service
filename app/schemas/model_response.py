from typing import List, Optional
from pydantic import BaseModel

class ModelResponse(BaseModel):
    """Response model for object detection predictions.
    
    This model standardizes the format of detection results across different YOLO models.
    It supports both standard and oriented bounding boxes (OBB).
    
    Attributes:
        bboxes (List[List[float]]): List of bounding boxes, where each box is represented as
                                  [x, y, width, height] in normalized coordinates (0-1).
        scores (List[float]): Confidence scores for each detection, ranging from 0 to 1.
        thetas (Optional[List[float]]): Rotation angles in radians for oriented bounding boxes.
                                     For standard bounding boxes, this will be None or a list of zeros.
        class_names (Optional[List[str]]): Human-readable class names for each detection.
                                         Will be None if class names are not available.
        class_ids (Optional[List[int]]): Numeric class IDs for each detection.
                                      Will be None if class IDs are not available.
    """
    bboxes: List[List[float]]
    scores: List[float]
    thetas: Optional[List[float]] = None
    class_names: Optional[List[str]] = None
    class_ids: Optional[List[int]] = None

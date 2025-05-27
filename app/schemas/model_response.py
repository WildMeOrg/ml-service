from typing import List, Optional
from pydantic import BaseModel

class ModelResponse(BaseModel):
    bboxes: List[List[float]]  # List of lists for bbox values in x, y, w, h format
    scores: List[float]  # List of confidence values
    thetas: Optional[List[float]] = None  # Optional float for bbox rotation
    class_names: Optional[List[str]] = None  # Optional string for classified classname
    class_ids: Optional[List[int]] = None  # Optional integer for classified class id

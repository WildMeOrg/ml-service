from .base_model import BaseModel
import logging
from transformers import AutoModel
from fastapi import HTTPException
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MiewidModel(BaseModel):
    def __init__(self):
        self.model = None
        self.model_info = {}

    def load(self, device: str = "cuda", **kwargs) -> None:
        if kwargs['version'] == 3:
            model_tag = f"conservationxlabs/miewid-msv3"
            self.model = AutoModel.from_pretrained(model_tag, trust_remote_code=True)
        else:
            raise HTTPException(status_code=400, detail="Unsupported miewid version.")

        self.model.eval()
        self.model.to(device)

        self.model_info = {
            'model_type': 'miewid',
            'device': device,
            'imgsz': kwargs.get('imgsz', 440)
        }
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self.model_info
    
    def predict(self, **kwargs):
        raise HTTPException(status_code=400, detail=f"MiewID should not be used for prediction")

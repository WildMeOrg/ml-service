from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all model implementations."""
    
    @abstractmethod
    def load(self, model_path: str, device: str, **kwargs) -> None:
        """Load the model from the specified path.
        
        Args:
            model_path: Path to the model file or model identifier
            device: Device to load the model on (e.g., 'cpu', 'cuda')
            **kwargs: Additional model-specific parameters
        """
        pass
    
    @abstractmethod
    def predict(self, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Run inference on the provided image.
        
        Args:
            image_bytes: Image data as bytes
            **kwargs: Additional inference parameters
            
        Returns:
            Dictionary containing the prediction results
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        pass

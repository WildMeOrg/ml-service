import importlib
import logging
from typing import Dict, Any, Optional, Type
from .base_model import BaseModel

logger = logging.getLogger(__name__)

# Mapping of model types to their respective module and class names
MODEL_REGISTRY = {
    'yolo-ultralytics': {
        'module': 'app.models.yolo_ultralytics',
        'class': 'YOLOUltralyticsModel'
    },
    'megadetector': {
        'module': 'app.models.megadetector',
        'class': 'MegaDetectorModel',
    },
    'miewid': {
        'module': 'app.models.miewid',
        'class': 'MiewidModel'
    }
    # Add new model types here as they are implemented
}

class ModelHandler:
    """Handler class for managing multiple model instances.
    
    This class provides a unified interface to load and access different types of models.
    It maintains a registry of available model types and their implementations.
    """
    
    def __init__(self):
        """Initialize the ModelHandler with an empty models dictionary."""
        self.models = {}
    
    def load_model(self, model_id: str, model_type: str, model_path: str = "", 
                  device: str = 'cpu', **kwargs) -> None:
        """Load a model with the specified parameters.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of the model (must be registered in MODEL_REGISTRY)
            model_path: Path to the model file or model identifier
            device: Device to load the model on (e.g., 'cpu', 'cuda')
            **kwargs: Additional model-specific parameters
            
        Raises:
            ValueError: If the model type is not supported or if there's an error loading the model
        """
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Available types: {list(MODEL_REGISTRY.keys())}")
        
        try:
            # Dynamically import the model class
            model_info = MODEL_REGISTRY[model_type]
            module = importlib.import_module(model_info['module'])
            model_class = getattr(module, model_info['class'])
            
            # Create and initialize the model
            model_instance: BaseModel = model_class()
            model_instance.load(model_path=model_path, device=device, model_id=model_id, **kwargs)
            
            # Store the model instance
            self.models[model_id] = {
                'instance': model_instance,
                'type': model_type,
                'config': {
                    'model_path': model_path,
                    'device': device,
                    **kwargs
                }
            }
            
            logger.info(f"Successfully loaded model '{model_id}' of type '{model_type}'")
            
        except Exception as e:
            logger.error(f"Error loading model '{model_id}': {str(e)}")
            raise ValueError(f"Failed to load model '{model_id}': {str(e)}")
    
    def get_model(self, model_id: str) -> Optional[BaseModel]:
        """Get a loaded model by its ID.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            The model instance if found, None otherwise
        """
        return self.models.get(model_id, {}).get('instance')
    
    def predict(self, model_id: str, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Run inference using the specified model.
        
        Args:
            model_id: ID of the model to use for prediction
            image_bytes: Image data as bytes
            **kwargs: Additional inference parameters
            
        Returns:
            Dictionary containing the prediction results
            
        Raises:
            ValueError: If the model is not found
        """
        model_info = self.models.get(model_id)
        if not model_info:
            raise ValueError(f"Model with ID '{model_id}' not found")
            
        return model_info['instance'].predict(image_bytes, **kwargs)
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a loaded model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary containing model information
            
        Raises:
            ValueError: If the model is not found
        """
        model_info = self.models.get(model_id)
        if not model_info:
            raise ValueError(f"Model with ID '{model_id}' not found")
            
        return {
            'model_id': model_id,
            'model_type': model_info['type'],
            'config': model_info['config'],
            'info': model_info['instance'].get_model_info()
        }
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded models.
        
        Returns:
            Dictionary mapping model IDs to their information
        """
        return {
            model_id: self.get_model_info(model_id)
            for model_id in self.models
        }

import json
import os
import sys
import logging
import argparse
from fastapi import FastAPI
from app.routers import predict_router
from app.models.model_handler import ModelHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Parse command line arguments
parser = argparse.ArgumentParser(description='FastAPI Model Serving Application')
parser.add_argument('--device', type=str, default='cpu', 
                   help='Device to run the models on (e.g., cpu, cuda, mps)')
parser.add_argument('--host', type=str, default='0.0.0.0', 
                   help='Host to run the server on')
parser.add_argument('--port', type=int, default=8000, 
                   help='Port to run the server on')
parser.add_argument('--reload', action='store_true', 
                   help='Enable auto-reload')
parser.add_argument('--workers', type=int, default=1, 
                   help='Number of worker processes')
args = parser.parse_args()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port, 
               reload=args.reload, workers=args.workers)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup.
    
    This function is called when the FastAPI application starts up. It performs the following actions:
    1. Creates a ModelHandler instance
    2. Loads the model configuration from model_config.json
    3. Initializes and loads all configured models with the specified device
    
    The models are stored in the application state and can be accessed via request.app.state.model_handler.
    """
    # Initialize model handler
    model_handler = ModelHandler()
    
    try:
        # Load model configuration
        config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        logger.info(f"Loading models on device: {args.device}")

        # Load each model from the configuration
        for model_cfg in config["models"]:
            model_id = model_cfg["model_id"]
            model_type = model_cfg["model_type"]
            
            # Extract model-specific parameters, excluding model_id and model_type
            model_params = {k: v for k, v in model_cfg.items() 
                          if k not in ["model_id", "model_type"]}
            
            # Load the model using the handler
            model_handler.load_model(
                model_id=model_id,
                model_type=model_type,
                device=args.device,
                **model_params
            )
            
            logger.info(f"Successfully loaded model: {model_id} ({model_type})")

        # Store the model handler in the application state
        app.state.model_handler = model_handler
        logger.info("All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        raise

app.include_router(predict_router.router)
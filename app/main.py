import json
import sys
import logging
from fastapi import FastAPI
from app.routers import predict_router, explain_router
from app.utils.yolo_handler import YOLOHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

import argparse

parser = argparse.ArgumentParser(description='FastAPI YOLO Application')
parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (e.g., cpu, cuda, mps)')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
args = parser.parse_args()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload, workers=args.workers)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup.
    
    This function is called when the FastAPI application starts up. It performs the following actions:
    1. Creates a YOLO handler instance
    2. Loads the model configuration from model_config.json
    3. Initializes and loads all configured models with the specified device
    
    The models are stored in the application state and can be accessed via request.app.state.yolo_handler.
    """
    yolo_handler = YOLOHandler()
    
    import os
    config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info("Server has started")

    device = args.device

    for model_cfg in config["models"]:
        yolo_handler.load_model(
            model_id=model_cfg["model_id"],
            model_path=model_cfg["model_path"],
            imgsz=model_cfg["imgsz"],
            conf=model_cfg["conf"],
            device=device
        )
    
    app.state.yolo_handler = yolo_handler

app.include_router(predict_router.router)
app.include_router(explain_router.router)

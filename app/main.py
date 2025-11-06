import json
import os
import sys
import logging
import argparse
from fastapi import FastAPI
from app.routers import predict_router, explain_router, extract_router, classify_router, pipeline_router
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
parser.add_argument('--device', type=str, default='cuda', 
                   help='Device to run the models on (e.g., cpu, cuda, mps)')
parser.add_argument('--host', type=str, default='0.0.0.0', 
                   help='Host to run the server on')
parser.add_argument('--port', type=int, default=8888, 
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
    app.state.device = args.device

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
        
        # Check if /data/db is accessible
        db_path = "/data/db"
        if not os.path.isdir(db_path) or not os.access(db_path, os.W_OK):
            logger.warning(f"Warning: Database directory {db_path} is not accessible or writable. This may cause issues with accessing image data.")
        
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint for Grafana and autoheal monitoring.
    
    Performs comprehensive checks:
    - Service is running
    - CUDA/GPU accessibility via nvidia-smi
    - GPU device detection
    - Model handler availability
    
    Returns:
        dict: Detailed health status with 200 OK response
    Raises:
        500: If critical health checks fail
    """
    import subprocess
    import torch
    from fastapi import HTTPException
    
    health_status = {
        "status": "healthy",
        "service": "running",
        "checks": {}
    }
    
    # Check CUDA via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_lines = result.stdout.strip().split('\n')
            health_status["checks"]["nvidia_smi"] = "ok"
            health_status["checks"]["gpu_count"] = len(gpu_lines)
            health_status["checks"]["gpus"] = [line.strip() for line in gpu_lines]
        else:
            health_status["checks"]["nvidia_smi"] = "failed"
            health_status["status"] = "unhealthy"
            raise HTTPException(status_code=500, detail="nvidia-smi check failed")
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        health_status["checks"]["nvidia_smi"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=500, detail=f"GPU check failed: {str(e)}")
    
    # Check PyTorch CUDA availability
    try:
        cuda_available = torch.cuda.is_available()
        health_status["checks"]["torch_cuda"] = "available" if cuda_available else "unavailable"
        if cuda_available:
            health_status["checks"]["torch_gpu_count"] = torch.cuda.device_count()
        else:
            health_status["status"] = "unhealthy"
            raise HTTPException(status_code=500, detail="PyTorch CUDA unavailable")
    except Exception as e:
        health_status["checks"]["torch_cuda"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=500, detail=f"PyTorch CUDA check failed: {str(e)}")
    
    # Check if models are loaded
    try:
        if hasattr(app.state, 'model_handler') and app.state.model_handler:
            model_count = len(app.state.model_handler.models)
            health_status["checks"]["models_loaded"] = model_count
        else:
            health_status["checks"]["models_loaded"] = 0
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["models_loaded"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

app.include_router(predict_router.router)
app.include_router(explain_router.router)
app.include_router(extract_router.router)
app.include_router(classify_router.router)
app.include_router(pipeline_router.router)

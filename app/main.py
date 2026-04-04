import json
import os
import sys
import logging
import argparse

# Add vendor/ to path for vendored dependencies (e.g. lightnet)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vendor'))
from fastapi import FastAPI
from app.routers import predict_router, explain_router, extract_router, classify_router, pipeline_router, assign_router, wbia_compat_router
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
        
        # Check if /data/db is accessible (read-only mount from host for shared image paths)
        db_path = "/data/db"
        if os.path.isdir(db_path) and os.access(db_path, os.R_OK):
            logger.info(f"Shared image directory {db_path} is accessible")
        else:
            logger.info(f"Shared image directory {db_path} not mounted (WBIA image path sharing disabled)")
        
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up models and free GPU memory on shutdown."""
    import torch
    if hasattr(app.state, 'model_handler'):
        for model_id, info in app.state.model_handler.models.items():
            info['instance'].model = None
        torch.cuda.empty_cache()
        logger.info("Models unloaded and GPU memory freed")

@app.get("/health")
async def health_check():
    """Health check endpoint for Grafana and autoheal monitoring.

    Device-aware: GPU checks only run when device is cuda.
    Returns 200 if healthy, 500 if critical checks fail.
    """
    import subprocess
    import torch
    from fastapi import HTTPException

    device = getattr(app.state, 'device', 'cpu')

    health_status = {
        "status": "healthy",
        "service": "running",
        "device": device,
        "checks": {}
    }

    # GPU checks only when running on CUDA
    if device == "cuda":
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
        except HTTPException:
            raise
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            health_status["checks"]["nvidia_smi"] = f"error: {str(e)}"
            health_status["status"] = "unhealthy"
            raise HTTPException(status_code=500, detail=f"GPU check failed: {str(e)}")

        try:
            cuda_available = torch.cuda.is_available()
            health_status["checks"]["torch_cuda"] = "available" if cuda_available else "unavailable"
            if cuda_available:
                health_status["checks"]["torch_gpu_count"] = torch.cuda.device_count()
            else:
                health_status["status"] = "unhealthy"
                raise HTTPException(status_code=500, detail="PyTorch CUDA unavailable")
        except HTTPException:
            raise
        except Exception as e:
            health_status["checks"]["torch_cuda"] = f"error: {str(e)}"
            health_status["status"] = "unhealthy"
            raise HTTPException(status_code=500, detail=f"PyTorch CUDA check failed: {str(e)}")
    else:
        health_status["checks"]["gpu"] = "skipped (device is not cuda)"

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
app.include_router(assign_router.router)
app.include_router(wbia_compat_router.router)

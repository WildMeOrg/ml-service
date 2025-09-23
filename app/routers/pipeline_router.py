import logging
from fastapi import APIRouter, HTTPException, Request, status, Depends
from typing import Dict, Any, List, Optional
import httpx
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from app.models.model_handler import ModelHandler
from app.models.efficientnet import EfficientNetModel
from app.models.miewid import MiewidModel
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# Limit concurrent pipeline operations to prevent OOM errors
MAX_CONCURRENT_PIPELINES = 1
pipeline_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)

def is_url(string: str) -> bool:
    """Check if a string is a valid URL.
    
    Args:
        string: The string to check
        
    Returns:
        bool: True if the string is a URL, False otherwise
    """
    return string.startswith(('http://', 'https://'))

class PipelineRequest(BaseModel):
    """Request model for pipeline endpoint."""
    predict_model_id: str = Field(..., description="ID of the model to use for prediction (bbox detection)")
    classify_model_id: str = Field(..., description="ID of the EfficientNet model to use for classification")
    extract_model_id: str = Field(..., description="ID of the MiewID model to use for embeddings extraction")
    image_uri: str = Field(..., description="URI of the image to process (URL or file path)")
    bbox_score_threshold: float = Field(default=0.5, description="Minimum bbox score threshold to process")
    predict_model_params: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional parameters to override prediction model configuration"
    )

async def get_model_handler(request: Request) -> ModelHandler:
    """Dependency to get the model handler from the app state."""
    return request.app.state.model_handler

@router.post("/", response_model=Dict[str, Any])
async def run_pipeline(
    pipeline_request: PipelineRequest,
    handler: ModelHandler = Depends(get_model_handler)
):
    """Run the complete pipeline: predict -> classify + extract for each bbox above threshold.
    
    Args:
        pipeline_request: The pipeline request containing all model IDs and parameters
        handler: The model handler instance
        
    Returns:
        Dictionary containing the pipeline results with bboxes, classifications, and embeddings
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    async with pipeline_semaphore:
        try:
            # Validate all models exist and are of correct types
            predict_model = handler.get_model(pipeline_request.predict_model_id)
            classify_model = handler.get_model(pipeline_request.classify_model_id)
            extract_model = handler.get_model(pipeline_request.extract_model_id)
            
            available_models = list(handler.list_models().keys())
            
            if not predict_model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": f"Prediction model '{pipeline_request.predict_model_id}' not found.",
                        "available_models": available_models
                    }
                )
            
            if not classify_model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": f"Classification model '{pipeline_request.classify_model_id}' not found.",
                        "available_models": available_models
                    }
                )
            
            if not extract_model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": f"Extraction model '{pipeline_request.extract_model_id}' not found.",
                        "available_models": available_models
                    }
                )
            
            # Validate model types
            if not isinstance(classify_model, EfficientNetModel):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{pipeline_request.classify_model_id}' is not an EfficientNet model. Only EfficientNet models support classification."
                )
            
            if not isinstance(extract_model, MiewidModel):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{pipeline_request.extract_model_id}' is not a MiewID model. Only MiewID models support embeddings extraction."
                )
            
            # Download image if it's a URL
            if is_url(pipeline_request.image_uri):
                async with httpx.AsyncClient() as client:
                    response = await client.get(pipeline_request.image_uri)
                    response.raise_for_status()
                    image_bytes = response.content
            else:
                # Handle local file path
                file_path = Path(pipeline_request.image_uri)
                if not file_path.exists():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File not found: {pipeline_request.image_uri}"
                    )
                with open(file_path, "rb") as f:
                    image_bytes = f.read()
            
            # Step 1: Run prediction to get bboxes
            logger.info(f"Running prediction with model {pipeline_request.predict_model_id}")
            
            # Get model info for default parameters
            model_info = handler.get_model_info(pipeline_request.predict_model_id)
            
            # Prepare prediction parameters, overriding defaults with any provided parameters
            predict_params = model_info['config'].copy()
            if pipeline_request.predict_model_params:
                predict_params.update(pipeline_request.predict_model_params)
            
            # Remove any parameters that shouldn't be passed to predict
            predict_params.pop('model_path', None)
            predict_params.pop('device', None)
            
            # Run prediction in a thread pool
            predict_result = await run_in_threadpool(
                predict_model.predict,
                image_bytes=image_bytes,
                **predict_params
            )
            
            # Step 2: Filter bboxes by score threshold
            filtered_bboxes = []
            
            # Handle different prediction result formats
            if 'bboxes' in predict_result and 'scores' in predict_result:
                # YOLO format: separate arrays for bboxes, scores, class_ids, etc.
                bboxes = predict_result.get('bboxes', [])
                scores = predict_result.get('scores', [])
                class_ids = predict_result.get('class_ids', [])
                class_names = predict_result.get('class_names', [])
                thetas = predict_result.get('thetas', [])
                
                for i, (bbox, score) in enumerate(zip(bboxes, scores)):
                    if score >= pipeline_request.bbox_score_threshold:
                        filtered_bboxes.append({
                            'bbox': bbox,
                            'score': score,
                            'class_id': class_ids[i] if i < len(class_ids) else None,
                            'class': class_names[i] if i < len(class_names) else None,
                            'theta': thetas[i] if i < len(thetas) else 0.0
                        })
            elif 'predictions' in predict_result:
                # Standard format: list of prediction objects
                for prediction in predict_result['predictions']:
                    if prediction.get('score', 0) >= pipeline_request.bbox_score_threshold:
                        filtered_bboxes.append(prediction)
            
            logger.info(f"Found {len(filtered_bboxes)} bboxes above threshold {pipeline_request.bbox_score_threshold}")
            
            # Step 3: Run classification and extraction for each filtered bbox
            pipeline_results = []
            original_classify_results = []
            original_extract_results = []
            
            for i, bbox_prediction in enumerate(filtered_bboxes):
                bbox_coords = bbox_prediction.get('bbox', [])
                if len(bbox_coords) != 4:
                    logger.warning(f"Skipping bbox {i}: invalid coordinates {bbox_coords}")
                    continue
                
                # YOLO already returns bbox in [x, y, width, height] format, no conversion needed
                x, y, width, height = bbox_coords
                bbox_list = [int(x), int(y), int(width), int(height)]
                
                # Validate bbox coordinates
                if width <= 0 or height <= 0:
                    logger.warning(f"Skipping bbox {i}: invalid dimensions width={width}, height={height}")
                    continue
                
                # Run classification and extraction in parallel for this bbox
                classify_task = run_in_threadpool(
                    classify_model.predict,
                    image_bytes=image_bytes,
                    bbox=bbox_list,
                    theta=0.0
                )
                
                extract_task = run_in_threadpool(
                    extract_model.extract_embeddings,
                    image_bytes=image_bytes,
                    bbox=tuple(bbox_list),
                    theta=0.0
                )
                
                # Wait for both tasks to complete
                classify_result, embeddings = await asyncio.gather(classify_task, extract_task)
                
                # Store original results for backup
                original_classify_results.append(classify_result)
                original_extract_results.append({
                    'embeddings': embeddings.tolist(),
                    'embeddings_shape': list(embeddings.shape),
                    'bbox': bbox_list
                })
                
                # Extract the top classification result
                top_classification = None
                if 'predictions' in classify_result and classify_result['predictions']:
                    top_class = classify_result['predictions'][0]
                    top_classification = {
                        'class': top_class.get('label'),  # EfficientNet uses 'label' not 'class'
                        'probability': top_class.get('probability'),
                        'class_id': top_class.get('index')  # EfficientNet uses 'index' not 'class_id'
                    }
                
                # Create clean pipeline result entry
                bbox_result = {
                    'bbox': bbox_coords,
                    'bbox_score': bbox_prediction.get('score'),
                    'detection_class': bbox_prediction.get('class'),
                    'detection_class_id': bbox_prediction.get('class_id'),
                    'classification': top_classification,
                    'embedding': embeddings.tolist(),
                    'embedding_shape': list(embeddings.shape)
                }
                
                pipeline_results.append(bbox_result)
            
            # Calculate total predictions based on format
            total_predictions = 0
            if 'bboxes' in predict_result:
                total_predictions = len(predict_result.get('bboxes', []))
            elif 'predictions' in predict_result:
                total_predictions = len(predict_result.get('predictions', []))
            
            # Prepare final response
            final_result = {
                'image_uri': pipeline_request.image_uri,
                'models_used': {
                    'predict_model_id': pipeline_request.predict_model_id,
                    'classify_model_id': pipeline_request.classify_model_id,
                    'extract_model_id': pipeline_request.extract_model_id
                },
                'bbox_score_threshold': pipeline_request.bbox_score_threshold,
                'total_predictions': total_predictions,
                'filtered_predictions': len(filtered_bboxes),
                'pipeline_results': pipeline_results,
                'original_predict': predict_result,
                'original_classify': original_classify_results,
                'original_extract': original_extract_results
            }
            
            return final_result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Error downloading image: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error downloading image: {str(e)}"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Pipeline error: {str(e)}"
            )

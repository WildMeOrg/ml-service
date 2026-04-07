import logging
from fastapi import APIRouter, HTTPException, Request, status, Depends
from typing import Dict, Any, List, Optional
import httpx
import asyncio
from pydantic import BaseModel, Field
from app.models.model_handler import ModelHandler
from app.models.efficientnet import EfficientNetModel
from app.models.miewid import MiewidModel
from app.models.densenet_orientation import DenseNetOrientationModel
from app.utils.image_uri import resolve_image_uri, sanitize_uri_for_response, sanitize_uri_for_logging
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# Limit concurrent pipeline operations to prevent OOM errors
MAX_CONCURRENT_PIPELINES = 2
pipeline_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)

class PipelineRequest(BaseModel):
    """Request model for pipeline endpoint."""
    predict_model_id: str = Field(..., description="ID of the model to use for prediction (bbox detection)")
    classify_model_id: str = Field(..., description="ID of the EfficientNet model to use for classification")
    extract_model_id: str = Field(..., description="ID of the MiewID model to use for embeddings extraction")
    orientation_model_id: Optional[str] = Field(None, description="ID of the DenseNet orientation model (optional)")
    image_uri: str = Field(..., description="URI of the image to process (URL or file path)")
    bbox_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum bbox score threshold to process")
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

            # Validate optional orientation model
            orientation_model = None
            if pipeline_request.orientation_model_id:
                orientation_model = handler.get_model(pipeline_request.orientation_model_id)
                if not orientation_model:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail={
                            "error": f"Orientation model '{pipeline_request.orientation_model_id}' not found.",
                            "available_models": available_models
                        }
                    )
                if not isinstance(orientation_model, DenseNetOrientationModel):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Model '{pipeline_request.orientation_model_id}' is not a DenseNet orientation model."
                    )
            
            # Resolve image bytes from URI (URL, data URI, or local path)
            try:
                image_bytes = await resolve_image_uri(pipeline_request.image_uri)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
            
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
                theta = float(bbox_prediction.get('theta', 0.0))

                # Validate bbox coordinates
                if width <= 0 or height <= 0:
                    logger.warning(f"Skipping bbox {i}: invalid dimensions width={width}, height={height}")
                    continue

                # Run orientation, classification, and extraction in parallel
                tasks = []
                task_names = []

                classify_task = run_in_threadpool(
                    classify_model.predict,
                    image_bytes=image_bytes,
                    bbox=bbox_list,
                    theta=theta
                )
                tasks.append(classify_task)
                task_names.append('classify')

                extract_task = run_in_threadpool(
                    extract_model.extract_embeddings,
                    image_bytes=image_bytes,
                    bbox=tuple(bbox_list),
                    theta=theta
                )
                tasks.append(extract_task)
                task_names.append('extract')

                if orientation_model:
                    orientation_task = run_in_threadpool(
                        orientation_model.predict,
                        image_bytes=image_bytes,
                        bbox=bbox_list,
                        theta=theta
                    )
                    tasks.append(orientation_task)
                    task_names.append('orientation')

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Unpack results, handling per-task failures
                classify_result = results[0]
                embeddings = results[1]
                orientation_result = results[2] if len(results) > 2 else None

                if isinstance(classify_result, Exception):
                    logger.warning(f"Classification failed for bbox {i}: {classify_result}")
                    classify_result = {}
                if isinstance(embeddings, Exception):
                    logger.warning(f"Extraction failed for bbox {i}: {embeddings}")
                    embeddings = None
                if isinstance(orientation_result, Exception):
                    logger.warning(f"Orientation failed for bbox {i}: {orientation_result}")
                    orientation_result = None

                embeddings_list = embeddings.tolist() if embeddings is not None else None
                embeddings_shape = list(embeddings.shape) if embeddings is not None else None

                # Store original results
                original_classify_results.append(classify_result)
                original_extract_results.append({
                    'embeddings': embeddings_list,
                    'embeddings_shape': embeddings_shape,
                    'bbox': bbox_list
                })

                # Extract the top classification result
                top_classification = None
                if isinstance(classify_result, dict) and 'predictions' in classify_result and classify_result['predictions']:
                    top_class = classify_result['predictions'][0]
                    top_classification = {
                        'class': top_class.get('label'),
                        'probability': top_class.get('probability'),
                        'class_id': top_class.get('index')
                    }

                # Extract orientation top prediction
                top_orientation = None
                if isinstance(orientation_result, dict) and 'predictions' in orientation_result:
                    preds = orientation_result['predictions']
                    if preds:
                        top_orientation = {
                            'label': preds[0].get('label'),
                            'probability': preds[0].get('probability'),
                            'class_id': preds[0].get('index')
                        }

                # Create clean pipeline result entry
                bbox_result = {
                    'bbox': bbox_coords,
                    'theta': theta,
                    'bbox_score': bbox_prediction.get('score'),
                    'detection_class': bbox_prediction.get('class'),
                    'detection_class_id': bbox_prediction.get('class_id'),
                    'classification': top_classification,
                    'embedding': embeddings_list,
                    'embedding_shape': embeddings_shape
                }
                if top_orientation is not None:
                    bbox_result['orientation'] = top_orientation
                
                pipeline_results.append(bbox_result)
            
            # Calculate total predictions based on format
            total_predictions = 0
            if 'bboxes' in predict_result:
                total_predictions = len(predict_result.get('bboxes', []))
            elif 'predictions' in predict_result:
                total_predictions = len(predict_result.get('predictions', []))
            
            # Prepare final response
            final_result = {
                'image_uri': sanitize_uri_for_response(pipeline_request.image_uri),
                'models_used': {
                    'predict_model_id': pipeline_request.predict_model_id,
                    'classify_model_id': pipeline_request.classify_model_id,
                    'extract_model_id': pipeline_request.extract_model_id,
                    **(({'orientation_model_id': pipeline_request.orientation_model_id}
                        ) if pipeline_request.orientation_model_id else {})
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

import logging
import math
from fastapi import APIRouter, HTTPException, Request, status, Depends
from typing import Dict, Any, List, Optional
import httpx
import asyncio
from pydantic import BaseModel, Field
from app.models.model_handler import ModelHandler
from app.models.efficientnet import EfficientNetModel
from app.models.densenet_classifier import DenseNetClassifierModel
from app.models.densenet_wilddog_cascade import DenseNetWildDogCascadeModel
from app.models.miewid import MiewidModel
from app.models.densenet_orientation import DenseNetOrientationModel
from app.models.wbia_orientation import WbiaOrientationModel, OrientationInferenceError
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
    classify_model_id: str = Field(..., description="ID of the classification model (EfficientNet or DenseNet-classifier)")
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
            if not isinstance(classify_model, (EfficientNetModel, DenseNetClassifierModel,
                                              DenseNetWildDogCascadeModel)):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{pipeline_request.classify_model_id}' must be "
                           f"EfficientNet, DenseNet-classifier or DenseNet-wilddog-cascade "
                           f"for the classify slot."
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
                # Two orientation kinds, dispatched on type:
                #   DenseNetOrientationModel -> a viewpoint LABEL (legacy)
                #   WbiaOrientationModel     -> THETA (the rotation regressor)
                if not isinstance(orientation_model,
                                  (DenseNetOrientationModel, WbiaOrientationModel)):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Model '{pipeline_request.orientation_model_id}' is not a "
                               f"DenseNet orientation or wbia-orientation model."
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

            # Resolve extract-model version once for the response. The
            # Wildbook v2 contract requires embedding_model_id +
            # embedding_model_version on every result so consumers can
            # match against persisted embeddings without consulting a
            # second config source. Empty / None / "None" version values
            # are treated as missing and fall back to "1" so the response
            # never carries a literally-broken version string.
            extract_model_info = handler.get_model_info(pipeline_request.extract_model_id)
            extract_model_version = "1"
            if extract_model_info and isinstance(extract_model_info, dict):
                extract_cfg = extract_model_info.get('config') or {}
                raw_version = extract_cfg.get('version')
                if raw_version is not None:
                    version_str = str(raw_version).strip()
                    if version_str and version_str.lower() != 'none':
                        extract_model_version = version_str

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
            
            # Step 2b: orientation FIRST, batched, when a theta regressor is
            # configured. theta must be known before the crop is embedded, so this
            # cannot run alongside classify/extract in the gather below. One
            # predict_batch per image = 3 TTA forwards total, not 3 per bbox.
            #
            # FAIL-CLOSED, at REQUEST level: if orientation fails for ANY bbox we
            # return 500 and emit nothing. Falling back to the detector's theta
            # (lightnet always gives 0.0) would embed an unrotated crop -- exactly
            # the regression this model type exists to repair. A *predicted* 0.0 is
            # valid; a *missing* one is not, so it must not be defaulted. Partial
            # success is also rejected: silently dropping a detection is how
            # annotations go missing.
            theta_regressor = isinstance(orientation_model, WbiaOrientationModel)
            wbia_orientation_results = None
            if theta_regressor and filtered_bboxes:
                ori_bboxes = [bp.get('bbox', []) for bp in filtered_bboxes]
                try:
                    wbia_orientation_results = await run_in_threadpool(
                        orientation_model.predict_batch,
                        image_bytes=image_bytes,
                        bboxes=ori_bboxes,
                    )
                except OrientationInferenceError as e:
                    logger.error(f"Orientation failed; failing the request: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Orientation model "
                               f"'{pipeline_request.orientation_model_id}' could not "
                               f"produce a trustworthy theta: {e}"
                    )
                # Ordered 1:1 is load-bearing -- a misaligned row would attach one
                # bbox's theta to another's crop.
                if len(wbia_orientation_results) != len(filtered_bboxes):
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Orientation returned "
                               f"{len(wbia_orientation_results)} result(s) for "
                               f"{len(filtered_bboxes)} bbox(es)."
                    )
                # Validate EVERY row here, not lazily inside the consumer loop.
                # Checking only the count and then dereferencing per-bbox would let
                # a malformed row N run classify/extract for rows 0..N-1 before the
                # request failed -- request-level fail-closed must mean no consumer
                # runs at all.
                for idx, ori in enumerate(wbia_orientation_results):
                    if not isinstance(ori, dict):
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Orientation result {idx} is not an object.")
                    th_val = ori.get('theta')
                    if not isinstance(th_val, (int, float)) or isinstance(th_val, bool) \
                            or not math.isfinite(float(th_val)):
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Orientation result {idx} has a non-finite or "
                                   f"missing theta: {th_val!r}")
                    eb = ori.get('effective_bbox')
                    if (not isinstance(eb, (list, tuple)) or len(eb) != 4
                            or not all(isinstance(v, int) and not isinstance(v, bool)
                                       for v in eb)):
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Orientation result {idx} has a malformed "
                                   f"effective_bbox: {eb!r}")

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
                theta_source = 'detector'
                detector_bbox = list(bbox_coords)   # audit: what the detector said

                if wbia_orientation_results is not None:
                    ori = wbia_orientation_results[i]
                    theta = float(ori['theta'])
                    theta_source = 'orientation'
                    # effective_bbox is the slice orientation ACTUALLY used (NumPy
                    # slicing does not clamp, and a degenerate crop falls back to
                    # the full frame). classify, extract and the emitted result
                    # must all use it, or theta describes a region other than the
                    # crop it rotates.
                    bbox_list = list(ori['effective_bbox'])
                    bbox_coords = bbox_list
                elif bbox_list[2] <= 0 or bbox_list[3] <= 0:
                    # Guard AFTER integerization: a raw width of 0.5 passes
                    # `width <= 0` but int(0.5) == 0, so consumers would receive a
                    # degenerate box. Only skip when orientation is NOT resolving
                    # bboxes for us -- its degenerate-crop fallback needs to see these.
                    logger.warning(f"Skipping bbox {i}: invalid dimensions "
                                   f"width={width}, height={height} "
                                   f"(integerized to {bbox_list[2]}x{bbox_list[3]})")
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

                if orientation_model and not theta_regressor:
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
                    # Embedding is required by the Wildbook v2 response
                    # contract (every result must carry a non-empty
                    # `embedding` array). A null/missing embedding would
                    # make the entire response invalid for v2 consumers,
                    # so fail the request rather than emit a poisoned
                    # success body. Classification/orientation can still
                    # soft-fail because they are optional in the contract.
                    logger.error(f"Extraction failed for bbox {i}: {embeddings}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Embedding extraction failed for bbox {i}: {embeddings}"
                    )
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
                top_species = None
                top_viewpoint = None
                if isinstance(classify_result, dict) and 'predictions' in classify_result and classify_result['predictions']:
                    top_class = classify_result['predictions'][0]
                    top_classification = {
                        'class': top_class.get('label'),
                        'probability': top_class.get('probability'),
                        'class_id': top_class.get('index')
                    }
                    top_species = top_class.get('species')
                    top_viewpoint = top_class.get('viewpoint')

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

                # Flatten the embedding to a 1D list. MiewID returns a
                # [1, D] tensor per crop; .tolist() gives [[d1, d2, ...]].
                # Wildbook v2 expects a flat array of doubles directly on
                # the result entry.
                flat_embedding = None
                if embeddings_list is not None:
                    if embeddings_list and isinstance(embeddings_list[0], list):
                        flat_embedding = embeddings_list[0]
                    else:
                        flat_embedding = embeddings_list

                # Create clean pipeline result entry
                bbox_result = {
                    'bbox': bbox_coords,
                    'detector_bbox': detector_bbox,
                    'theta': theta,
                    'theta_source': theta_source,
                    'bbox_score': bbox_prediction.get('score'),
                    'detection_class': bbox_prediction.get('class'),
                    'detection_class_id': bbox_prediction.get('class_id'),
                    'classification': top_classification,
                    'embedding': flat_embedding,
                    'embedding_shape': embeddings_shape,
                    # Wildbook v2 contract: embedding_model_id +
                    # embedding_model_version must live on each result so
                    # the persisted Embedding row's (method, version) pair
                    # matches what the matching code looks up later.
                    'embedding_model_id': pipeline_request.extract_model_id,
                    'embedding_model_version': extract_model_version,
                }
                if top_orientation is not None:
                    bbox_result['orientation'] = top_orientation
                if top_species is not None:
                    bbox_result['iaClass'] = top_species
                if top_viewpoint is not None:
                    bbox_result['viewpoint'] = top_viewpoint

                pipeline_results.append(bbox_result)
            
            # Calculate total predictions based on format
            total_predictions = 0
            if 'bboxes' in predict_result:
                total_predictions = len(predict_result.get('bboxes', []))
            elif 'predictions' in predict_result:
                total_predictions = len(predict_result.get('predictions', []))
            
            # Prepare final response.
            #
            # Wildbook v2 contract:
            #   - top-level `success: True` (validated by MlServiceClient)
            #   - top-level `results` array (renamed from pipeline_results)
            # Both are required by validatePipelineResponse in MlServiceClient.
            # `pipeline_results` is kept as an alias for one release so any
            # in-flight non-Wildbook callers (test scripts, etc.) don't break.
            final_result = {
                'success': True,
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
                'results': pipeline_results,
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

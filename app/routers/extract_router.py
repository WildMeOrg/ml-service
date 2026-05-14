import logging
from fastapi import APIRouter, HTTPException, Request, status, Depends
from typing import Dict, Any, List, Optional, Union
import httpx
import asyncio
from pydantic import BaseModel, Field, model_validator
from app.models.model_handler import ModelHandler
from app.models.miewid import MiewidModel
from app.utils.image_uri import resolve_image_uri, sanitize_uri_for_response
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extract", tags=["Embeddings Extraction"])

# Limit concurrent extractions to prevent OOM errors
MAX_CONCURRENT_EXTRACTIONS = 2
extract_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTIONS)

class ExtractRequest(BaseModel):
    """Request model for embeddings extraction endpoint.

    Accepts both legacy `model_id` and Wildbook-v2 `extract_model_id` for
    the model identifier. Whichever is provided is normalized into
    `model_id` by the pre-validator below.

    `bbox` accepts integers OR floats. Wildbook v2's MlServiceClient sends
    doubles; legacy scripts send ints. Either is fine.
    """
    model_id: Optional[str] = Field(None, description="ID of the MiewID model to use for extraction (legacy field name)")
    extract_model_id: Optional[str] = Field(None, description="ID of the MiewID model (Wildbook v2 field name; alias for model_id)")
    image_uri: str = Field(..., description="URI of the image to process (URL or file path)")
    bbox: Optional[List[float]] = Field(None, description="Optional bounding box coordinates [x, y, width, height]. If not provided, uses full image")
    theta: float = Field(default=0.0, description="Rotation angle in radians")

    @model_validator(mode='after')
    def _normalize_model_id(self):
        if not self.model_id and not self.extract_model_id:
            raise ValueError("either 'model_id' or 'extract_model_id' must be provided")
        # Prefer extract_model_id when both are set (v2 client should win
        # over any accidentally-stale legacy field on the same request).
        if self.extract_model_id:
            self.model_id = self.extract_model_id
        return self

async def get_model_handler(request: Request) -> ModelHandler:
    """Dependency to get the model handler from the app state."""
    return request.app.state.model_handler

@router.post("/", response_model=Dict[str, Any])
async def extract_embeddings(
    extract_request: ExtractRequest,
    handler: ModelHandler = Depends(get_model_handler)
):
    """Extract embeddings from an image using MiewID model with bounding box and rotation.
    
    Args:
        extract_request: The extraction request containing model_id, image_uri, bbox, and theta
        handler: The model handler instance
        
    Returns:
        Dictionary containing the embeddings and metadata
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    async with extract_semaphore:
        try:
            # Validate bbox format if provided
            if extract_request.bbox is not None and len(extract_request.bbox) != 4:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Bounding box must contain exactly 4 values: [x, y, width, height]"
                )
            
            # Get the model instance
            model = handler.get_model(extract_request.model_id)
            if not model:
                available_models = list(handler.list_models().keys())
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": f"Model '{extract_request.model_id}' not found.",
                        "available_models": available_models
                    }
                )
            
            # Check if the model is a MiewID model
            if not isinstance(model, MiewidModel):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{extract_request.model_id}' is not a MiewID model. Only MiewID models support embeddings extraction."
                )
            
            # Resolve image bytes from URI (URL, data URI, or local path)
            try:
                image_bytes = await resolve_image_uri(extract_request.image_uri)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
                with open(file_path, "rb") as f:
                    image_bytes = f.read()
            
            # Convert bbox to tuple of ints if provided. The underlying
            # extract_embeddings expects integer pixel coordinates; v2
            # senders may pass doubles (e.g. 12.0), so round down.
            bbox_tuple = (
                tuple(int(v) for v in extract_request.bbox)
                if extract_request.bbox is not None else None
            )

            # Extract embeddings in a thread pool
            embeddings = await run_in_threadpool(
                model.extract_embeddings,
                image_bytes=image_bytes,
                bbox=bbox_tuple,
                theta=extract_request.theta
            )

            # MiewID returns shape [1, D]; flatten to a 1D list for the
            # response so Wildbook v2 sees `embedding: [...]` rather than
            # `embedding: [[...]]`.
            embeddings_list = embeddings.tolist()
            if embeddings_list and isinstance(embeddings_list[0], list):
                flat_embedding = embeddings_list[0]
            else:
                flat_embedding = embeddings_list

            # Resolve extract-model version for the response so consumers
            # can match against persisted embeddings by (method, version).
            # Empty / None / "None" version values are treated as missing
            # and fall back to "1" so the response never carries a
            # literally-broken version string.
            extract_model_info = handler.get_model_info(extract_request.model_id)
            extract_model_version = "1"
            if extract_model_info and isinstance(extract_model_info, dict):
                extract_cfg = extract_model_info.get('config') or {}
                raw_version = extract_cfg.get('version')
                if raw_version is not None:
                    version_str = str(raw_version).strip()
                    if version_str and version_str.lower() != 'none':
                        extract_model_version = version_str

            # Prepare response.
            #
            # Wildbook v2 contract:
            #   - top-level `success: True`
            #   - top-level `embedding` (singular, flat array of doubles)
            #   - `embedding_model_id` + `embedding_model_version`
            # Legacy keys (`embeddings`, `embeddings_shape`, `model_id`)
            # are kept so existing test scripts continue working.
            result = {
                'success': True,
                'model_id': extract_request.model_id,
                'embedding': flat_embedding,
                'embedding_model_id': extract_request.model_id,
                'embedding_model_version': extract_model_version,
                'embeddings': embeddings_list,
                'embeddings_shape': list(embeddings.shape),
                'bbox': extract_request.bbox,
                'theta': extract_request.theta,
                'image_uri': sanitize_uri_for_response(extract_request.image_uri)
            }

            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Error downloading image: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error downloading image: {str(e)}"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Embeddings extraction error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Embeddings extraction error: {str(e)}"
            )

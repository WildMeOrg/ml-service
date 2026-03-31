"""Router for part-body assignment endpoint.

Matches part annotations (e.g. lion+head) to body annotations using
geometric features and species-specific scikit-learn classifiers.
"""

import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.models.assigner import (
    AssignerHandler,
    SPECIES_CONFIG,
    compute_pair_features,
    make_assignments,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assign", tags=["Assignment"])


class Annotation(BaseModel):
    """A single annotation with bbox and metadata."""
    aid: int = Field(..., description="Annotation ID")
    bbox: List[float] = Field(..., description="Bounding box [x, y, width, height]")
    theta: float = Field(default=0.0, description="Rotation angle in radians")
    viewpoint: Optional[str] = Field(default=None, description="Viewpoint string (e.g. 'left', 'frontleft')")
    is_part: bool = Field(..., description="Whether this is a part annotation")
    species: Optional[str] = Field(default=None, description="Species of this annotation")


class AssignRequest(BaseModel):
    """Request model for the assignment endpoint."""
    species: str = Field(..., description="Primary species for classifier selection")
    annotations: List[Annotation] = Field(..., description="List of annotations to assign")
    image_width: int = Field(..., description="Width of the source image in pixels")
    image_height: int = Field(..., description="Height of the source image in pixels")
    cutoff_score: float = Field(default=0.5, description="Minimum score threshold for assignment")


@router.post("/", response_model=Dict[str, Any])
async def assign_parts(request: Request, body: AssignRequest):
    """Assign part annotations to body annotations using geometric features.

    Computes geometric features for all possible (part, body) pairs,
    scores them with a species-specific classifier, and greedily assigns
    the highest-scoring pairs.

    Args:
        request: The HTTP request
        body: The assignment request

    Returns:
        Dictionary with assigned pairs and unassigned annotation IDs
    """
    try:
        # Get or create assigner handler from app state
        if not hasattr(request.app.state, 'assigner_handler'):
            request.app.state.assigner_handler = AssignerHandler()
        handler = request.app.state.assigner_handler

        # Separate parts and bodies
        parts = [a for a in body.annotations if a.is_part]
        bodies = [a for a in body.annotations if not a.is_part]

        if not parts or not bodies:
            return {
                'assigned_pairs': [],
                'unassigned_aids': [a.aid for a in body.annotations],
            }

        # Validate bboxes
        for ann in body.annotations:
            if len(ann.bbox) != 4:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Annotation {ann.aid}: bbox must have 4 values [x, y, w, h]"
                )

        # Load classifier
        classifier = handler.get_classifier(body.species)
        feature_type = handler.get_feature_type(body.species)

        # Compute features for all (part, body) pairs
        pair_parts = []
        pair_bodies = []
        features = []

        for part in parts:
            for bod in bodies:
                pair_parts.append(part.aid)
                pair_bodies.append(bod.aid)

                feat = compute_pair_features(
                    part_bbox=part.bbox,
                    part_theta=part.theta,
                    part_viewpoint=part.viewpoint,
                    body_bbox=bod.bbox,
                    body_theta=bod.theta,
                    body_viewpoint=bod.viewpoint,
                    image_width=body.image_width,
                    image_height=body.image_height,
                    feature_type=feature_type,
                )
                features.append(feat)

        # Score pairs
        scores_raw = classifier.predict_proba(features)
        # predict_proba returns [P_false, P_true] — take P_true
        scores = [float(s[1]) for s in scores_raw]

        # Get species for validation
        part_species = [
            (p.species or body.species).split('+')[0] for p in parts
            for _ in bodies
        ]
        body_species = [
            (b.species or body.species).split('+')[0]
            for _ in parts
            for b in bodies
        ]

        # Run greedy assignment
        assigned_pairs, unassigned_aids = make_assignments(
            part_aids=pair_parts,
            body_aids=pair_bodies,
            scores=scores,
            cutoff_score=body.cutoff_score,
            supported_species=set(SPECIES_CONFIG.keys()),
            part_species=part_species,
            body_species=body_species,
        )

        return {
            'assigned_pairs': assigned_pairs,
            'unassigned_aids': unassigned_aids,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Assignment error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Assignment error: {str(e)}"
        )

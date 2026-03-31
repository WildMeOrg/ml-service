"""Assigner model for matching part annotations to body annotations.

Ports the WBIA assigner logic: computes geometric features from bbox pairs,
scores them with a species-specific scikit-learn classifier, and uses greedy
assignment to match parts to bodies.
"""

import math
import logging
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from joblib import load as joblib_load

from ..utils.checkpoint_utils import get_checkpoint_path

logger = logging.getLogger(__name__)

# Species-to-model mapping (matches WBIA SPECIES_CONFIG_MAP)
SPECIES_CONFIG = {
    'wild_dog': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.wd_v0.joblib',
        'feature_type': 'viewpoint',
    },
    'wild_dog_dark': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.wd_v0.joblib',
        'feature_type': 'viewpoint',
    },
    'wild_dog_light': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.wd_v0.joblib',
        'feature_type': 'viewpoint',
    },
    'wild_dog_puppy': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.wd_v0.joblib',
        'feature_type': 'viewpoint',
    },
    'wild_dog_standard': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.wd_v0.joblib',
        'feature_type': 'viewpoint',
    },
    'wild_dog_tan': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.wd_v0.joblib',
        'feature_type': 'viewpoint',
    },
    'chelonia_mydas': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.iot_dummies_v0.joblib',
        'feature_type': 'unit_viewpoint',
    },
    'eretmochelys_imbricata': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.iot_dummies_v0.joblib',
        'feature_type': 'unit_viewpoint',
    },
    'lepidochelys_olivacea': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.iot_dummies_v0.joblib',
        'feature_type': 'unit_viewpoint',
    },
    'turtle_green': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.iot_dummies_v0.joblib',
        'feature_type': 'unit_viewpoint',
    },
    'turtle_hawksbill': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.iot_dummies_v0.joblib',
        'feature_type': 'unit_viewpoint',
    },
    'turtle_oliveridley': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.iot_dummies_v0.joblib',
        'feature_type': 'unit_viewpoint',
    },
    'lion': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.lions_v0.joblib',
        'feature_type': 'unit_viewpoint',
    },
    'lioness': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.lions_v0.joblib',
        'feature_type': 'unit_viewpoint',
    },
    'lion_general': {
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.lions_v0.joblib',
        'feature_type': 'unit_viewpoint',
    },
}

FALLBACK_SPECIES = 'wild_dog'


def _viewpoint_to_lrudfb_bools(viewpoint: Optional[str]) -> List[bool]:
    """Convert a viewpoint string to left/right/up/down/front/back booleans."""
    if viewpoint is None:
        return [False] * 6
    vp = viewpoint.lower()
    return [
        'left' in vp,
        'right' in vp,
        'up' in vp,
        'down' in vp,
        'front' in vp,
        'back' in vp,
    ]


def _viewpoint_to_lrudfb_unit_vector(viewpoint: Optional[str]) -> List[float]:
    """Convert a viewpoint string to a unit-normalized LRUDFB float vector."""
    bools = _viewpoint_to_lrudfb_bools(viewpoint)
    floats = [float(b) for b in bools]
    length = math.sqrt(sum(bools))
    if length == 0:
        length = -1  # Match WBIA behavior: avoid division by zero
    return [f / length for f in floats]


def _bbox_to_polygon_verts(bbox: List[float], theta: float = 0.0) -> List[List[float]]:
    """Convert bbox [x, y, w, h] + theta to 4 rotated vertices."""
    x, y, w, h = bbox
    # Center of bbox
    cx = x + w / 2
    cy = y + h / 2
    # Unrotated corners relative to center
    corners = [
        (-w / 2, -h / 2),
        (w / 2, -h / 2),
        (w / 2, h / 2),
        (-w / 2, h / 2),
    ]
    if theta != 0.0:
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rotated = []
        for dx, dy in corners:
            rx = dx * cos_t - dy * sin_t + cx
            ry = dx * sin_t + dy * cos_t + cy
            rotated.append([rx, ry])
        return rotated
    else:
        return [[cx + dx, cy + dy] for dx, dy in corners]


def compute_pair_features(
    part_bbox: List[float], part_theta: float, part_viewpoint: Optional[str],
    body_bbox: List[float], body_theta: float, body_viewpoint: Optional[str],
    image_width: int, image_height: int,
    feature_type: str = 'unit_viewpoint',
) -> List[float]:
    """Compute geometric features for a single (part, body) annotation pair.

    Mirrors WBIA's assigner_viewpoint_unit_features / assigner_viewpoint_features.

    Args:
        part_bbox: [x, y, w, h] of the part annotation
        part_theta: rotation angle of part annotation
        part_viewpoint: viewpoint string (e.g. 'left', 'frontleft') or None
        body_bbox: [x, y, w, h] of the body annotation
        body_theta: rotation angle of body annotation
        body_viewpoint: viewpoint string or None
        image_width: width of the source image
        image_height: height of the source image
        feature_type: 'viewpoint' (bool) or 'unit_viewpoint' (unit vector)

    Returns:
        Feature vector (list of floats)
    """
    try:
        from shapely import geometry
    except ImportError:
        raise ImportError("shapely is required for assigner feature computation. Install with: pip install shapely")

    # Normalize bbox coordinates to [0, 1]
    norm_part_bbox = [
        part_bbox[0] / image_width, part_bbox[1] / image_height,
        part_bbox[2] / image_width, part_bbox[3] / image_height,
    ]
    norm_body_bbox = [
        body_bbox[0] / image_width, body_bbox[1] / image_height,
        body_bbox[2] / image_width, body_bbox[3] / image_height,
    ]

    # Get rotated vertices (normalized)
    part_verts = _bbox_to_polygon_verts(norm_part_bbox, part_theta)
    body_verts = _bbox_to_polygon_verts(norm_body_bbox, body_theta)

    part_poly = geometry.Polygon(part_verts)
    body_poly = geometry.Polygon(body_verts)

    intersect_poly = part_poly.intersection(body_poly)
    intersect_area = intersect_poly.area
    int_area_scalar = math.sqrt(intersect_area)

    part_area = norm_part_bbox[2] * norm_part_bbox[3]
    body_area = norm_body_bbox[2] * norm_body_bbox[3]
    union_area = part_area + body_area - intersect_area

    int_over_union = intersect_area / union_area if union_area > 0 else 0.0
    int_over_part = intersect_area / part_area if part_area > 0 else 0.0
    int_over_body = intersect_area / body_area if body_area > 0 else 0.0
    part_over_body = part_area / body_area if body_area > 0 else 0.0

    part_body_distance = part_poly.distance(body_poly)

    part_centroid = part_poly.centroid
    body_centroid = body_poly.centroid
    part_body_centroid_dist = part_centroid.distance(body_centroid)

    # Build feature vector
    # Part vertices (8 values) + part center (2) + body bbox (4) + body center (2)
    features = [
        part_verts[0][0], part_verts[0][1],
        part_verts[1][0], part_verts[1][1],
        part_verts[2][0], part_verts[2][1],
        part_verts[3][0], part_verts[3][1],
        part_centroid.x, part_centroid.y,
        norm_body_bbox[0], norm_body_bbox[1],
        norm_body_bbox[2], norm_body_bbox[3],
        body_centroid.x, body_centroid.y,
        int_area_scalar,
        part_body_distance,
        part_body_centroid_dist,
    ]

    if feature_type == 'unit_viewpoint':
        # Unit viewpoint version adds body_to_part vector
        body_to_part_x = part_centroid.x - body_centroid.x
        body_to_part_y = part_centroid.y - body_centroid.y
        features.extend([body_to_part_x, body_to_part_y])

    features.extend([int_over_union, int_over_part, int_over_body, part_over_body])

    # Viewpoint features
    if feature_type == 'unit_viewpoint':
        features.extend(_viewpoint_to_lrudfb_unit_vector(part_viewpoint))
        features.extend(_viewpoint_to_lrudfb_unit_vector(body_viewpoint))
    else:
        bools_part = _viewpoint_to_lrudfb_bools(part_viewpoint)
        bools_body = _viewpoint_to_lrudfb_bools(body_viewpoint)
        features.extend([float(b) for b in bools_part])
        features.extend([float(b) for b in bools_body])

    return features


class AssignerHandler:
    """Handles loading and caching of species-specific assigner classifiers."""

    def __init__(self):
        self._models: Dict[str, Any] = {}

    def get_classifier(self, species: str) -> Any:
        """Get the classifier for a species, loading and caching if needed."""
        # Resolve species to config
        base_species = species.split('+')[0]  # Strip part suffix
        if base_species not in SPECIES_CONFIG:
            if FALLBACK_SPECIES in SPECIES_CONFIG:
                logger.warning(
                    f"No assigner model for species '{base_species}', "
                    f"falling back to '{FALLBACK_SPECIES}'"
                )
                base_species = FALLBACK_SPECIES
            else:
                raise ValueError(f"No assigner model for species '{base_species}'")

        model_url = SPECIES_CONFIG[base_species]['model_url']

        # Check cache (keyed by URL since multiple species share models)
        if model_url in self._models:
            return self._models[model_url]

        # Download and load
        model_path = get_checkpoint_path(model_url)
        clf = joblib_load(model_path)
        self._models[model_url] = clf
        logger.info(f"Loaded assigner classifier for '{base_species}' from {model_url}")
        return clf

    def get_feature_type(self, species: str) -> str:
        """Get the feature type for a species."""
        base_species = species.split('+')[0]
        if base_species in SPECIES_CONFIG:
            return SPECIES_CONFIG[base_species]['feature_type']
        return SPECIES_CONFIG.get(FALLBACK_SPECIES, {}).get('feature_type', 'unit_viewpoint')


def make_assignments(
    part_aids: List[int],
    body_aids: List[int],
    scores: List[float],
    cutoff_score: float = 0.5,
    supported_species: Optional[set] = None,
    part_species: Optional[List[str]] = None,
    body_species: Optional[List[str]] = None,
) -> Tuple[List[Dict], List[int]]:
    """Greedy assignment algorithm matching parts to bodies.

    Mirrors WBIA's _make_assignments().

    Args:
        part_aids: List of part annotation IDs
        body_aids: List of body annotation IDs (parallel to part_aids)
        scores: Classifier scores for each (part, body) pair
        cutoff_score: Minimum score threshold
        supported_species: Set of species with assigner models
        part_species: Species of each part (for validation)
        body_species: Species of each body (for validation)

    Returns:
        Tuple of (assigned_pairs, unassigned_aids)
    """
    # Zero out scores for mismatched species support
    if supported_species and part_species and body_species:
        for i in range(len(scores)):
            ps = part_species[i] in supported_species
            bs = body_species[i] in supported_species
            # NXOR: both supported or both unsupported is OK
            if ps != bs:
                scores[i] = 0.0

    # Sort by score descending
    sorted_pairs = sorted(
        zip(part_aids, body_aids, scores),
        key=lambda x: x[2],
        reverse=True,
    )

    assigned_pairs = []
    assigned_parts = set()
    assigned_bodies = set()
    n_true_pairs = min(len(set(part_aids)), len(set(body_aids)))

    for part_aid, body_aid, score in sorted_pairs:
        if score < cutoff_score:
            break

        if part_aid not in assigned_parts and body_aid not in assigned_bodies:
            assigned_pairs.append({
                'part_aid': part_aid,
                'body_aid': body_aid,
                'score': score,
            })
            assigned_parts.add(part_aid)
            assigned_bodies.add(body_aid)

        if len(assigned_parts) >= n_true_pairs or len(assigned_bodies) >= n_true_pairs:
            break

    all_aids = set(part_aids) | set(body_aids)
    assigned_aid_set = assigned_parts | assigned_bodies
    unassigned_aids = sorted(list(all_aids - assigned_aid_set))

    return assigned_pairs, unassigned_aids

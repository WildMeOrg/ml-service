"""Tests for the assigner model and endpoint.

Verifies:
- Geometric feature computation
- Viewpoint boolean/unit vector conversion
- Greedy assignment algorithm
- Species fallback behavior
"""

import math
import pytest

from app.models.assigner import (
    _viewpoint_to_lrudfb_bools,
    _viewpoint_to_lrudfb_unit_vector,
    _bbox_to_polygon_verts,
    compute_pair_features,
    make_assignments,
    SPECIES_CONFIG,
    FALLBACK_SPECIES,
)


class TestViewpointConversion:

    def test_lrudfb_bools_left(self):
        result = _viewpoint_to_lrudfb_bools('left')
        assert result == [True, False, False, False, False, False]

    def test_lrudfb_bools_frontleft(self):
        result = _viewpoint_to_lrudfb_bools('frontleft')
        assert result == [True, False, False, False, True, False]

    def test_lrudfb_bools_none(self):
        result = _viewpoint_to_lrudfb_bools(None)
        assert result == [False] * 6

    def test_unit_vector_single(self):
        result = _viewpoint_to_lrudfb_unit_vector('left')
        assert abs(result[0] - 1.0) < 1e-6  # left=1/sqrt(1)=1
        assert all(abs(v) < 1e-6 for v in result[1:])

    def test_unit_vector_two_directions(self):
        result = _viewpoint_to_lrudfb_unit_vector('frontleft')
        expected = 1.0 / math.sqrt(2)
        assert abs(result[0] - expected) < 1e-6  # left
        assert abs(result[4] - expected) < 1e-6  # front

    def test_unit_vector_none(self):
        """None viewpoint should return all zeros (with -1 divisor to avoid div by zero)."""
        result = _viewpoint_to_lrudfb_unit_vector(None)
        # All should be 0.0 / -1 = -0.0 which == 0.0
        assert all(abs(v) < 1e-6 for v in result)


class TestBboxToPolygon:

    def test_no_rotation(self):
        verts = _bbox_to_polygon_verts([10, 20, 30, 40], theta=0.0)
        assert len(verts) == 4
        # Check corners of bbox [10, 20, 30, 40] → center (25, 40)
        assert abs(verts[0][0] - 10) < 1e-6  # x
        assert abs(verts[0][1] - 20) < 1e-6  # y
        assert abs(verts[2][0] - 40) < 1e-6  # x + w
        assert abs(verts[2][1] - 60) < 1e-6  # y + h

    def test_with_rotation(self):
        """Rotation should change vertex positions."""
        verts_0 = _bbox_to_polygon_verts([0, 0, 10, 10], theta=0.0)
        verts_r = _bbox_to_polygon_verts([0, 0, 10, 10], theta=math.pi / 4)
        # Rotated vertices should differ
        assert verts_0 != verts_r


class TestFeatureComputation:

    def test_overlapping_bboxes(self):
        """Two overlapping bboxes should have nonzero IoU."""
        features = compute_pair_features(
            part_bbox=[10, 10, 50, 50], part_theta=0.0, part_viewpoint='left',
            body_bbox=[20, 20, 60, 60], body_theta=0.0, body_viewpoint='right',
            image_width=100, image_height=100,
            feature_type='unit_viewpoint',
        )
        # Features should be a list of floats
        assert all(isinstance(f, float) for f in features)
        # Feature 21 is int_over_union — should be > 0 for overlapping bboxes
        assert features[21] > 0

    def test_non_overlapping_bboxes(self):
        """Non-overlapping bboxes should have zero IoU."""
        features = compute_pair_features(
            part_bbox=[0, 0, 10, 10], part_theta=0.0, part_viewpoint=None,
            body_bbox=[80, 80, 10, 10], body_theta=0.0, body_viewpoint=None,
            image_width=100, image_height=100,
            feature_type='viewpoint',
        )
        # IoU should be 0
        assert features[19] == 0.0  # int_over_union for viewpoint type

    def test_viewpoint_feature_length(self):
        """Viewpoint type should produce 35 features (23 geo + 12 bool)."""
        features = compute_pair_features(
            part_bbox=[10, 10, 30, 30], part_theta=0.0, part_viewpoint='left',
            body_bbox=[20, 20, 40, 40], body_theta=0.0, body_viewpoint='right',
            image_width=100, image_height=100,
            feature_type='viewpoint',
        )
        # 8 (part verts) + 2 (part center) + 4 (body bbox) + 2 (body center)
        # + 1 (int_area_scalar) + 1 (distance) + 1 (centroid_dist)
        # + 4 (IoU, int_over_part, int_over_body, part_over_body)
        # + 6 (part viewpoint) + 6 (body viewpoint) = 35
        assert len(features) == 35

    def test_unit_viewpoint_feature_length(self):
        """Unit viewpoint type should produce 37 features (25 geo + 12 unit)."""
        features = compute_pair_features(
            part_bbox=[10, 10, 30, 30], part_theta=0.0, part_viewpoint='left',
            body_bbox=[20, 20, 40, 40], body_theta=0.0, body_viewpoint='right',
            image_width=100, image_height=100,
            feature_type='unit_viewpoint',
        )
        # Same as viewpoint but +2 for body_to_part_x/y = 37
        assert len(features) == 37


class TestMakeAssignments:

    def test_basic_assignment(self):
        """High-scoring pair should be assigned."""
        assigned, unassigned = make_assignments(
            part_aids=[1, 1],
            body_aids=[2, 3],
            scores=[0.9, 0.3],
            cutoff_score=0.5,
        )
        assert len(assigned) == 1
        assert assigned[0]['part_aid'] == 1
        assert assigned[0]['body_aid'] == 2
        assert 3 in unassigned

    def test_greedy_no_duplicate(self):
        """Each part/body should only be assigned once."""
        assigned, unassigned = make_assignments(
            part_aids=[1, 1, 2, 2],
            body_aids=[3, 4, 3, 4],
            scores=[0.95, 0.9, 0.85, 0.8],
            cutoff_score=0.5,
        )
        # Best: (1,3,0.95), then (2,4,0.8) since 3 is taken
        assert len(assigned) == 2
        part_ids = {p['part_aid'] for p in assigned}
        body_ids = {p['body_aid'] for p in assigned}
        assert part_ids == {1, 2}
        assert body_ids == {3, 4}

    def test_below_cutoff(self):
        """Pairs below cutoff should not be assigned."""
        assigned, unassigned = make_assignments(
            part_aids=[1],
            body_aids=[2],
            scores=[0.3],
            cutoff_score=0.5,
        )
        assert len(assigned) == 0
        assert set(unassigned) == {1, 2}

    def test_empty_input(self):
        """Empty lists should return empty results."""
        assigned, unassigned = make_assignments(
            part_aids=[],
            body_aids=[],
            scores=[],
        )
        assert assigned == []
        assert unassigned == []

    def test_species_mismatch_zeroed(self):
        """Mismatched species support should zero out scores."""
        assigned, unassigned = make_assignments(
            part_aids=[1],
            body_aids=[2],
            scores=[0.9],
            cutoff_score=0.5,
            supported_species={'lion'},
            part_species=['lion'],
            body_species=['unknown_species'],
        )
        # Score should be zeroed since one is supported and the other isn't
        assert len(assigned) == 0


class TestSpeciesConfig:

    def test_fallback_species_exists(self):
        assert FALLBACK_SPECIES in SPECIES_CONFIG

    def test_all_configs_have_required_keys(self):
        for species, config in SPECIES_CONFIG.items():
            assert 'model_url' in config, f"{species} missing model_url"
            assert 'feature_type' in config, f"{species} missing feature_type"
            assert config['feature_type'] in ('viewpoint', 'unit_viewpoint')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

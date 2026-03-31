"""Tests for configurable EfficientNet model.

Verifies that:
- Default label map works (backward compatibility)
- Custom label maps from config work
- Compound label parsing (species:viewpoint) works
- Multi-label vs single-label modes work
- WBIA checkpoint format (state + classes) is loaded correctly
"""

import json
import os
import tempfile
import numpy as np
import torch
from unittest.mock import patch, MagicMock

import pytest

from app.models.efficientnet import EfficientNetModel, ImgClassifier, DEFAULT_LABEL_MAP


def _make_fake_checkpoint(classes=None, use_wbia_format=True):
    """Create a fake checkpoint file for testing."""
    model = ImgClassifier(model_arch='tf_efficientnet_b4_ns', n_class=len(classes or DEFAULT_LABEL_MAP))
    if use_wbia_format and classes:
        checkpoint = {
            'state': model.state_dict(),
            'classes': classes,
        }
    else:
        checkpoint = model.state_dict()

    path = tempfile.mktemp(suffix='.pth')
    torch.save(checkpoint, path)
    return path


class TestEfficientNetConfig:

    def test_default_label_map(self):
        """Existing models with no label_map config should use DEFAULT_LABEL_MAP."""
        path = _make_fake_checkpoint(list(DEFAULT_LABEL_MAP.values()), use_wbia_format=False)
        try:
            model = EfficientNetModel()
            model.load(checkpoint_path=path, device='cpu', model_id='test-default')
            assert model.label_map == DEFAULT_LABEL_MAP
            assert model.multi_label is True
            assert model.parse_compound_labels is False
        finally:
            os.unlink(path)

    def test_wbia_checkpoint_classes(self):
        """WBIA checkpoints with 'classes' key should auto-populate label_map."""
        classes = ['chelonia_mydas:left', 'chelonia_mydas:right', 'eretmochelys_imbricata:left']
        path = _make_fake_checkpoint(classes, use_wbia_format=True)
        try:
            model = EfficientNetModel()
            model.load(checkpoint_path=path, device='cpu', model_id='test-wbia')
            assert len(model.label_map) == 3
            assert model.label_map[0] == 'chelonia_mydas:left'
            assert model.label_map[2] == 'eretmochelys_imbricata:left'
        finally:
            os.unlink(path)

    def test_explicit_label_map_overrides_checkpoint(self):
        """Config label_map should take priority over checkpoint classes."""
        classes = ['a', 'b', 'c']
        path = _make_fake_checkpoint(classes, use_wbia_format=True)
        explicit_map = {'0': 'alpha', '1': 'beta', '2': 'gamma'}
        try:
            model = EfficientNetModel()
            model.load(
                checkpoint_path=path, device='cpu', model_id='test-override',
                label_map=explicit_map,
            )
            assert model.label_map == {0: 'alpha', 1: 'beta', 2: 'gamma'}
        finally:
            os.unlink(path)

    def test_compound_label_parsing(self):
        """parse_compound_labels=True should split labels on ':' in predictions."""
        classes = ['species_a:left', 'species_b:right']
        path = _make_fake_checkpoint(classes, use_wbia_format=True)
        try:
            model = EfficientNetModel()
            model.load(
                checkpoint_path=path, device='cpu', model_id='test-compound',
                parse_compound_labels=True, multi_label=False,
            )

            # Create a test image (100x100 white)
            import cv2
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            _, img_bytes = cv2.imencode('.png', img)

            result = model.predict(image_bytes=img_bytes.tobytes())
            assert 'predictions' in result
            # With parse_compound_labels, each prediction should have species and viewpoint
            for pred in result['predictions']:
                if ':' in pred['label']:
                    assert 'species' in pred
                    assert 'viewpoint' in pred
        finally:
            os.unlink(path)

    def test_softmax_single_label(self):
        """multi_label=False should use softmax and return single top prediction."""
        classes = ['up', 'down', 'left']
        path = _make_fake_checkpoint(classes, use_wbia_format=True)
        try:
            model = EfficientNetModel()
            model.load(
                checkpoint_path=path, device='cpu', model_id='test-softmax',
                multi_label=False,
            )

            import cv2
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            _, img_bytes = cv2.imencode('.png', img)

            result = model.predict(image_bytes=img_bytes.tobytes())
            assert len(result['predictions']) == 1
            # Probabilities should sum to ~1 for softmax
            total = sum(result['all_probabilities'])
            assert abs(total - 1.0) < 0.01
        finally:
            os.unlink(path)

    def test_module_prefix_stripping(self):
        """DataParallel 'module.' prefix in state dict should be stripped."""
        classes = ['a', 'b']
        base_model = ImgClassifier(model_arch='tf_efficientnet_b4_ns', n_class=2)
        # Add module. prefix to simulate DataParallel
        state_dict = {'module.' + k: v for k, v in base_model.state_dict().items()}
        checkpoint = {'state': state_dict, 'classes': classes}

        path = tempfile.mktemp(suffix='.pth')
        torch.save(checkpoint, path)
        try:
            model = EfficientNetModel()
            model.load(checkpoint_path=path, device='cpu', model_id='test-parallel')
            assert model.model is not None
        finally:
            os.unlink(path)

    def test_get_model_info_includes_new_fields(self):
        """get_model_info should report multi_label and parse_compound_labels."""
        path = _make_fake_checkpoint(list(DEFAULT_LABEL_MAP.values()), use_wbia_format=False)
        try:
            model = EfficientNetModel()
            model.load(
                checkpoint_path=path, device='cpu', model_id='test-info',
                multi_label=False, parse_compound_labels=True,
            )
            info = model.get_model_info()
            assert info['multi_label'] is False
            assert info['parse_compound_labels'] is True
        finally:
            os.unlink(path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

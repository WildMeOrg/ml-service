"""Regression: EfficientNet's compound-label path must use the shared
parser, which suppresses sentinel prefixes like 'species'. Before this fix,
EfficientNet would set entry['species'] = 'species' (literal namespace) for
the deployed efficientnet-classifier whose labels are 'species:up' etc."""
import os
import tempfile

import cv2
import numpy as np
import pytest
import torch

from app.models.efficientnet import EfficientNetModel, ImgClassifier


def _make_fake_checkpoint(classes):
    """Create a minimal fake checkpoint with the given class list."""
    base_model = ImgClassifier(model_arch='tf_efficientnet_b4_ns', n_class=len(classes))
    checkpoint = {'state': base_model.state_dict(), 'classes': classes}
    path = tempfile.mktemp(suffix='.pth')
    torch.save(checkpoint, path)
    return path


def test_efficientnet_sentinel_suppression():
    """When a prediction's label is 'species:up' and the model has
    parse_compound_labels=True, the per-prediction `species` field must be
    None (sentinel suppression), not the literal string 'species'."""
    classes = ['species:up', 'species:down']
    path = _make_fake_checkpoint(classes)
    try:
        model = EfficientNetModel()
        model.load(
            checkpoint_path=path,
            device='cpu',
            model_id='test-sentinel',
            parse_compound_labels=True,
            multi_label=False,
        )

        # Create a minimal white test image.
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, img_bytes = cv2.imencode('.png', img)

        result = model.predict(image_bytes=img_bytes.tobytes())
        assert 'predictions' in result
        assert len(result['predictions']) >= 1

        top = result['predictions'][0]
        # The label must be preserved as-is.
        assert ':' in top['label'], f"Expected compound label, got {top['label']!r}"
        # The 'species' key must be absent or None — 'species' is a sentinel prefix.
        assert top.get('species') is None, (
            f"expected sentinel 'species' suppressed (species=None), "
            f"got {top.get('species')!r}"
        )
        # Viewpoint must be the suffix after the colon.
        assert top.get('viewpoint') in ('up', 'down'), (
            f"expected viewpoint 'up' or 'down', got {top.get('viewpoint')!r}"
        )
    finally:
        os.unlink(path)

import io
import json
import sys
import types

import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image

from scripts.preflight import run_gate as gate


def encoded(mode):
    im = Image.new(mode, (8, 8))
    im.putpixel((0, 0), {'L': 73, 'RGBA': (10, 20, 30, 0), 'RGB': (10, 20, 30), 'I;16': 4096}[mode])
    data = io.BytesIO()
    im.save(data, format='PNG')
    return data.getvalue()


@pytest.mark.parametrize('mode,pixel', [('L', (73, 73, 73)), ('RGBA', (10, 20, 30))])
def test_wrapper_has_independent_rgb_copy(mode, pixel):
    original = encoded(mode)
    converted = gate.reference_image_bytes(original, 'canonicalization_wrapper')
    with Image.open(io.BytesIO(converted)) as rgb:
        assert rgb.mode == 'RGB'
        assert rgb.size == (8, 8)
        assert rgb.getpixel((0, 0)) == pixel
    with Image.open(io.BytesIO(original)) as source:
        assert source.mode == mode


def test_rgb_fidelity_bytes_are_unchanged():
    data = encoded('RGB')
    assert gate.reference_image_bytes(data, 'rgb_jpeg_downscale') is data


def test_regular_fidelity_rejects_grayscale():
    with pytest.raises(ValueError, match='rgb_jpeg_downscale is RGB-only'):
        gate.reference_image_bytes(encoded('L'), 'rgb_jpeg_downscale')


@pytest.mark.parametrize('mode', ['RGB', 'I;16'])
def test_wrapper_rejects_unsupported_modes(mode):
    with pytest.raises(ValueError, match='requires 8-bit'):
        gate.reference_image_bytes(encoded(mode), 'canonicalization_wrapper')


def test_grayscale_exif_uses_inference_decoder_orientation():
    from app.models.wbia_orientation import _canonicalize_rgb
    im = Image.fromarray(np.arange(45, dtype=np.uint8).reshape(5, 9))
    exif = Image.Exif()
    exif[274] = 6
    stream = io.BytesIO()
    im.save(stream, format='JPEG', exif=exif)
    original = stream.getvalue()
    converted = gate.reference_image_bytes(original, 'canonicalization_wrapper')
    np.testing.assert_array_equal(imageio.imread(io.BytesIO(converted)),
                                  _canonicalize_rgb(imageio.imread(io.BytesIO(original))))


@pytest.mark.parametrize('mode', ['L', 'RGBA'])
def test_gate_keeps_original_for_port_and_converts_only_reference(mode, tmp_path, monkeypatch):
    original = encoded(mode)
    (tmp_path / 'image.png').write_bytes(original)
    weights = tmp_path / 'weights.pth'
    weights.write_bytes(b'fake')
    calls = []

    class Reference:
        def __init__(self, *args, **kwargs):
            pass

        def theta(self, data, bbox):
            calls.append('reference')
            with Image.open(io.BytesIO(data)) as rgb, Image.open(io.BytesIO(original)) as source:
                assert rgb.mode == 'RGB'
                np.testing.assert_array_equal(np.asarray(rgb), np.asarray(source.convert('RGB')))
            return 0.0, [0.5] * 5

    class Port:
        def load(self, **kwargs):
            pass

        def predict_batch(self, data, bboxes):
            calls.append('port')
            assert data == original
            return [{'theta': 0.0, 'coords_normalized': [0.5] * 5, 'effective_bbox': bboxes[0]}]

    monkeypatch.setitem(sys.modules, 'reference_runner', types.SimpleNamespace(Reference=Reference))
    monkeypatch.setitem(sys.modules, 'app.models.wbia_orientation', types.SimpleNamespace(WbiaOrientationModel=Port))
    monkeypatch.setattr(gate, 'environment', lambda: {})
    manifest = {'thresholds': {'theta_circular_max_rad': 1e-5, 'coords_elementwise_max': 1e-6},
                'checkpoints': [{'model_id': 'test', 'path': str(weights)}],
                'fixtures': [{'file': 'image.png', 'bbox': [0, 0, 8, 8], 'stratum': 'canonicalization_wrapper'}],
                'strata': {'canonicalization_wrapper': {'min_samples': 1}}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    assert gate.main(['--manifest', str(path), '--fixtures', str(tmp_path),
                      '--artifact', str(tmp_path / 'artifact.json')]) == 0
    assert calls == ['reference', 'port']

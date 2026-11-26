import subprocess
import sys
from pathlib import Path

import pytest

from scripts.preflight import reference_runner as runner
from scripts.preflight.run_gate import (
    reference_image, resolve_reference_root, wbia_localization_boundary)


MODULES = ('config/default.py', 'models/cls_hrnet.py', 'utils/utils.py', 'core/evaluate.py')


def checkout(path):
    for name in MODULES:
        module = path / 'wbia_orientation' / name
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text(f'SOURCE = {str(path)!r}\n')
    return path


def test_reference_root_precedence():
    assert resolve_reference_root({}) == '/reference'
    manifest = {'reference_source': {'path': '/manifest'}}
    assert resolve_reference_root(manifest) == '/manifest'
    assert resolve_reference_root(manifest, '/cli') == '/cli'


def test_loads_configured_checkout(tmp_path):
    source = runner.load_reference(checkout(tmp_path))
    assert set(source) == {'cfg', 'hrnet', 'utils', 'eval'}
    assert all(module.SOURCE == str(tmp_path) for module in source.values())


def test_missing_checkout_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='--reference-root'):
        runner.load_reference(tmp_path)


def test_help_works_without_reference_checkout():
    script = Path(__file__).resolve().parents[1] / 'scripts/preflight/run_gate.py'
    result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--reference-root' in result.stdout


def test_roots_are_isolated_and_canonical_root_is_cached(tmp_path):
    a = runner.load_reference(checkout(tmp_path / 'a'))
    b = runner.load_reference(checkout(tmp_path / 'b'))
    assert a is runner.load_reference(tmp_path / 'a' / '.')
    assert a['cfg'].__name__ != b['cfg'].__name__
    assert a['cfg'].SOURCE != b['cfg'].SOURCE


def test_failed_import_cleans_up_module_namespace(tmp_path):
    checkout(tmp_path)
    (tmp_path / 'wbia_orientation/models/cls_hrnet.py').write_text('raise RuntimeError("broken")')
    before = {name for name in sys.modules if name.startswith('wd_')}
    with pytest.raises(RuntimeError, match='broken'):
        runner.load_reference(tmp_path)
    assert {name for name in sys.modules if name.startswith('wd_')} == before



@pytest.mark.parametrize("bbox,expected", [
    ([22, -28, 2839, 1492], [22, 0, 2838, 1464]),   # Sharkbook guam2 overhang
    ([10, 10, 50, 50], [10, 10, 50, 50]),
    ([-20, 10, 500, 50], [0, 10, 480, 50]),
    ([3000, 10, 50, 50], [2860, 10, 0, 50]),          # fully outside -> empty
])
def test_reference_gets_the_wbia_clipped_box(bbox, expected):
    """The reference plugin only ever saw boxes WBIA's localizations step had
    clipped, so the gate hands it the same, not the raw detection."""
    assert wbia_localization_boundary(bbox, 2860, 1464) == expected



def test_reference_gets_the_upright_image_wbia_stored_for_a_tagged_fixture():
    """WBIA rewrote EXIF-rotated images upright on import, so the reference
    plugin read upright pixels. The gate hands it the same (lossless PNG of an
    independent cv2 decode) and reports the UPRIGHT size for clipping."""
    import io
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    from tests.test_exif_orientation_decode import (
        _jpeg_with_orientation, _sensor_pixels)

    tagged = _jpeg_with_orientation(_sensor_pixels(), 6)          # sensor 120x200
    data, w, h = reference_image(tagged)
    assert (w, h) == (120, 200)                                     # portrait, upright
    got = np.asarray(imageio.imread(io.BytesIO(data)))
    expected = cv2.cvtColor(cv2.imdecode(np.frombuffer(tagged, np.uint8),
                                         cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    assert np.array_equal(got, expected)


def test_untagged_fixture_reaches_the_reference_byte_for_byte():
    from tests.test_exif_orientation_decode import _sensor_pixels
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(_sensor_pixels()).save(buf, format="JPEG")
    data, w, h = reference_image(buf.getvalue())
    assert data == buf.getvalue() and (w, h) == (200, 120)

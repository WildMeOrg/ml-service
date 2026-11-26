import subprocess
import sys
from pathlib import Path

import pytest

from scripts.preflight import reference_runner as runner
from scripts.preflight.run_gate import (
    reference_image, resolve_reference_root, stratum_shortfalls,
    wbia_localization_boundary)


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



@pytest.mark.parametrize("orientation", [2, 3, 4, 5, 6, 7, 8])
def test_reference_gets_the_upright_image_wbia_stored_for_a_tagged_jpeg(orientation):
    """WBIA rewrote EXIF-rotated images upright on import, so the reference
    plugin read upright pixels. The gate hands it the same (lossless PNG of a
    cv2 decode, as WBIA did) and reports the UPRIGHT size for clipping.
    Checked against the independent NumPy oracle, block-averaged because cv2
    and Pillow may round JPEG pixels differently."""
    import io
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    from tests.test_exif_orientation_decode import (
        ORACLE, _block_means, _jpeg_with_orientation, _sensor_pixels)

    tagged = _jpeg_with_orientation(_sensor_pixels(), orientation)       # sensor 120x200
    data, w, h, tag = reference_image(tagged)
    expected = ORACLE[orientation](np.array(Image.open(io.BytesIO(tagged)).convert("RGB")))
    got = np.asarray(imageio.imread(io.BytesIO(data)))
    assert tag == orientation
    assert (h, w) == expected.shape[:2]
    assert np.abs(_block_means(got) - _block_means(expected)).max() < 3


def test_untagged_fixture_reaches_the_reference_byte_for_byte():
    import io
    from PIL import Image
    from tests.test_exif_orientation_decode import _sensor_pixels
    buf = io.BytesIO()
    Image.fromarray(_sensor_pixels()).save(buf, format="JPEG")
    data, w, h, tag = reference_image(buf.getvalue())
    assert data == buf.getvalue() and (w, h) == (200, 120) and tag in (None, 1)


def test_tagged_non_jpeg_fixture_is_refused():
    """cv2 4.8 ignores WebP EXIF and 16-bit PNGs would lose depth: the EXIF
    parity oracle is only defined for JPEG."""
    import io
    import numpy as np
    from PIL import Image
    im = Image.fromarray(np.zeros((20, 30, 3), np.uint8))
    exif = im.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, format="PNG", exif=exif.tobytes())
    with pytest.raises(ValueError, match="JPEG"):
        reference_image(buf.getvalue())



def test_stratum_minimum_counts_distinct_fixtures_per_checkpoint():
    rows = [{"stratum": "exif_rotated", "checkpoint": c, "fixture": f}
            for c in ("a", "b", "c") for f in ("x.jpg", "y.jpg")]
    rows += [{"stratum": "exif_rotated", "checkpoint": "a", "fixture": "x.jpg"}]  # repeat
    out = stratum_shortfalls(rows, {"exif_rotated": {"min_samples": 3}}, ["a", "b", "c"])
    assert len(out) == 3 and all("2 fixtures < required 3" in m for m in out)
    assert stratum_shortfalls(rows, {"exif_rotated": {"min_samples": 2}}, ["a", "b", "c"]) == []


def test_compare_fixture_refuses_an_invalid_exif_tag_in_the_exif_stratum():
    import io
    import numpy as np
    from PIL import Image
    from scripts.preflight.run_gate import compare_fixture
    im = Image.fromarray(np.zeros((20, 30, 3), np.uint8))
    exif = im.getexif()
    exif[0x0112] = 9                       # not a valid orientation
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif.tobytes())
    rows, failures = [], []
    compare_fixture("m", {"file": "f.jpg", "stratum": "exif_rotated", "bbox": [0, 0, 5, 5]},
                    buf.getvalue(), ref=None, port=None, thr={}, rows=rows, failures=failures)
    assert rows == [] and "EXIF Orientation is 9" in failures[0]

#!/usr/bin/env python3
"""wbia-orientation fidelity gate — release-blocking host preflight.

Executes the pinned REFERENCE and the PORT live on the same bytes and compares
them to each other. There are no frozen expected values: they would pin one
implementation's output and rot on any dependency bump. See README.md.

Exit 0 = pass. Non-zero = do not deploy.
"""
import argparse, hashlib, json, math, os, sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))


def circular_error(a: float, b: float) -> float:
    d = a - b
    return abs(math.atan2(math.sin(d), math.cos(d)))


def environment() -> dict:
    import numpy, torch, timm, skimage, imageio, PIL
    return {"python": sys.version, "device": "cpu", "torch": torch.__version__, "timm": timm.__version__,
            "numpy": numpy.__version__, "scikit-image": skimage.__version__,
            "imageio": imageio.__version__, "pillow": PIL.__version__}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_reference_root(manifest, override=None):
    return override or manifest.get("reference_source", {}).get("path") or "/reference"


def reference_image_bytes(data, stratum):
    """Independently canonicalize only the reference's wrapper-fixture input.

    Use imageio for decoding, as both inference paths do (including EXIF
    handling), then Pillow for independent channel conversion. Restrict the
    wrapper fixtures to 8-bit L/RGBA to avoid lossy 16-bit conversions.
    """
    import io
    import imageio.v2 as imageio
    from PIL import Image

    with Image.open(io.BytesIO(data)) as original:
        if stratum != "canonicalization_wrapper":
            if original.mode != "RGB":
                raise ValueError(f"stratum {stratum} is RGB-only; fixture is mode {original.mode}")
            return data
        if original.mode not in ("L", "RGBA"):
            raise ValueError(
                "canonicalization_wrapper requires 8-bit grayscale (L) or RGBA; "
                f"got {original.mode}"
            )
    decoded = imageio.imread(io.BytesIO(data))
    if decoded.dtype.name != "uint8":
        raise ValueError("canonicalization_wrapper requires 8-bit decoded pixels")
    with Image.fromarray(decoded).convert("RGB") as rgb:
        output = io.BytesIO()
        rgb.save(output, format="PNG")
        return output.getvalue()


ERROR_LIMITS = ('theta_circular_max_rad', 'theta_circular_mean_rad',
                'coords_elementwise_max', 'coords_elementwise_mean')


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fixture_bboxes(fixture):
    if ('bbox' in fixture) == ('bboxes' in fixture):
        raise ValueError('provide exactly one of bbox or bboxes')
    boxes = [fixture['bbox']] if 'bbox' in fixture else fixture['bboxes']
    if not isinstance(boxes, list) or not boxes:
        raise ValueError('bboxes must be a nonempty list')
    if any(not isinstance(box, list) or len(box) != 4 or
           not all(finite_number(v) for v in box) for box in boxes):
        raise ValueError('each bbox must contain four finite numeric values')
    if fixture.get('stratum') == 'multi_detection':
        if len({tuple(map(int, box)) for box in boxes}) < 2:
            raise ValueError('multi_detection requires at least two distinct integerized bboxes')
    return boxes


def validate_result(result):
    if not isinstance(result, dict) or not finite_number(result.get('theta')):
        raise ValueError('result must contain a finite theta')
    coords = result.get('coords_normalized')
    if not isinstance(coords, (list, tuple)) or len(coords) != 5 or not all(map(finite_number, coords)):
        raise ValueError('result must contain exactly five finite coordinates')
    bbox = result.get('effective_bbox')
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or any(type(v) is not int for v in bbox):
        raise ValueError('result must contain four integer effective_bbox values')


def evaluate(manifest, fixtures_dir, reference_root, reference_cls, port_cls):
    rows, failures, summaries, checkpoints = [], [], {}, []
    thresholds = manifest.get('thresholds', {})
    for name in ERROR_LIMITS:
        if not finite_number(thresholds.get(name)) or thresholds[name] < 0:
            failures.append(f'threshold {name} must be finite and nonnegative')
    for name, expected in [('effective_bbox', 'exact'), ('predict_batch', 'exact count and order')]:
        if thresholds.get(name) != expected:
            failures.append(f'threshold contract {name} must be {expected!r}')
    if not manifest.get('checkpoints'):
        failures.append('checkpoints must be nonempty')
    if not manifest.get('fixtures'):
        failures.append('fixtures must be nonempty')
    strata = {k: v for k, v in manifest.get('strata', {}).items() if not k.startswith('_')}
    if not strata:
        failures.append('strata must be nonempty')
    for name, spec in strata.items():
        minimum = spec.get('min_samples')
        if type(minimum) is not int or minimum < 1:
            failures.append(f'stratum {name}: min_samples must be a positive integer')
    ids = [ck.get('model_id') for ck in manifest.get('checkpoints', [])]
    if any(not isinstance(model_id, str) or not model_id for model_id in ids) or len(set(ids)) != len(ids):
        failures.append('checkpoint model_id values must be nonempty and unique')
    if failures:
        return rows, failures, summaries, checkpoints

    for ck in manifest['checkpoints']:
        model_id = ck['model_id']
        model_rows = []
        coverage = {name: set() for name in strata}
        try:
            checkpoint_path = ck['path']
            checkpoint_digest = sha256(checkpoint_path)
            if ck.get('sha256') not in (None, '', '<record at preflight>', checkpoint_digest):
                raise ValueError('checkpoint hash mismatch (manifest identity broken)')
            checkpoint = {'model_id': model_id, 'path': checkpoint_path, 'sha256': checkpoint_digest}
            checkpoints.append(checkpoint)
            reference = reference_cls(checkpoint_path, reference_root=reference_root)
            checkpoint['reference_source'] = reference.source_identity
            port = port_cls()
            port.load(model_id=model_id, checkpoint_path=checkpoint_path, device='cpu')
            checkpoint['port_config'] = port.get_model_info()
        except Exception as exc:
            failures.append(f'{model_id}: checkpoint load failed: {exc}')
            continue

        for fixture in manifest['fixtures']:
            label = f"{model_id}/{fixture.get('file')}"
            print(f'Checking {label}', flush=True)
            try:
                stratum = fixture.get('stratum')
                if stratum not in strata:
                    raise ValueError(f'undeclared stratum {stratum!r}')
                if not isinstance(fixture.get('file'), str) or not fixture['file']:
                    raise ValueError("fixture must declare a nonempty 'file' path")
                boxes = fixture_bboxes(fixture)
                path = os.path.join(fixtures_dir, fixture['file'])
                with open(path, 'rb') as stream:
                    data = stream.read()
                digest = hashlib.sha256(data).hexdigest()
                if fixture.get('sha256') not in (None, '', '<fixture byte hash>', digest):
                    raise ValueError('fixture hash mismatch (manifest identity broken)')
                reference_data = reference_image_bytes(data, stratum)
                expected = [reference.predict(reference_data, box) for box in boxes]
                for result in expected:
                    validate_result(result)
                if stratum == 'multi_detection':
                    if len({tuple(r['effective_bbox']) for r in expected}) < 2:
                        raise ValueError('multi_detection requires distinct effective crops')
                    # Identical predictions cannot reveal a permuted batch. Require
                    # each pair to be distinguishable under the accepted tolerances.
                    for i, a in enumerate(expected):
                        for b in expected[i+1:]:
                            if (circular_error(a['theta'], b['theta']) <= 2 * thresholds['theta_circular_max_rad']
                                    and max(abs(x-y) for x, y in zip(a['coords_normalized'], b['coords_normalized']))
                                    <= 2 * thresholds['coords_elementwise_max']):
                                raise ValueError('multi_detection reference predictions cannot distinguish row order')
                actual = port.predict_batch(data, boxes)
                if not isinstance(actual, (list, tuple)) or len(actual) != len(boxes):
                    raise ValueError(f'predict_batch must return exactly {len(boxes)} rows')
                pending = []
                for index, (box, ref, result) in enumerate(zip(boxes, expected, actual)):
                    validate_result(result)
                    if list(ref['effective_bbox']) != list(result['effective_bbox']):
                        raise ValueError(f'row {index}: effective_bbox mismatch (crop or batch order)')
                    theta_error = circular_error(ref['theta'], result['theta'])
                    coord_errors = [abs(a-b) for a, b in zip(ref['coords_normalized'], result['coords_normalized'])]
                    if not all(map(finite_number, [theta_error, *coord_errors])):
                        raise ValueError('non-finite comparison errors')
                    pending.append({'checkpoint': model_id, 'fixture': fixture['file'],
                                    'fixture_sha256': digest, 'stratum': stratum, 'bbox': box, 'batch_index': index,
                                    'theta_ref': ref['theta'], 'theta_port': result['theta'],
                                    'coords_ref': ref['coords_normalized'], 'coords_port': result['coords_normalized'],
                                    'theta_err': theta_error, 'coord_err': max(coord_errors),
                                    'coord_errors': coord_errors, 'effective_bbox': result['effective_bbox'],
                                    'reference_effective_bbox': ref['effective_bbox']})
                model_rows.extend(pending)
                coverage[stratum].add((digest, tuple(sorted({tuple(r['effective_bbox']) for r in expected}))))
            except Exception as exc:
                failures.append(f'{label}: {exc}')

        for name, spec in strata.items():
            if len(coverage[name]) < spec['min_samples']:
                failures.append(f"{model_id}: stratum {name}: {len(coverage[name])} samples < required {spec['min_samples']}")
        if not model_rows:
            failures.append(f'{model_id}: no valid comparisons')
            continue
        theta_errors = [r['theta_err'] for r in model_rows]
        coord_errors = [error for r in model_rows for error in r['coord_errors']]
        metrics = {'theta_circular_max_rad': max(theta_errors),
                   'theta_circular_mean_rad': math.fsum(theta_errors) / len(theta_errors),
                   'coords_elementwise_max': max(coord_errors),
                   'coords_elementwise_mean': math.fsum(coord_errors) / len(coord_errors)}
        summaries[model_id] = {**metrics, 'comparisons': len(model_rows),
                               'coverage': {name: len(samples) for name, samples in coverage.items()}}
        for name, value in metrics.items():
            print(f'{model_id}: {name} {value:.3e} (limit {thresholds[name]:.3e})')
            if value > thresholds[name]:
                failures.append(f'{model_id}: {name} {value:.3e} > {thresholds[name]:.3e}')
        rows.extend(model_rows)
    return rows, failures, summaries, checkpoints


def write_artifact(path, artifact):
    """Never replace a previous artifact with partial JSON or nonfinite values."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(os.path.abspath(path)),
                                         suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            json.dump(artifact, stream, indent=2, allow_nan=False)
        # The container may run as root while release evidence is consumed
        # by the host user or CI. Artifacts are intentionally readable by both.
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--fixtures', required=True, help='directory of fixture images')
    ap.add_argument('--artifact', default='preflight-artifact.json')
    ap.add_argument('--reference-root', help='checkout root containing wbia_orientation')
    args = ap.parse_args(argv)
    started = time.monotonic()
    artifact = {'environment': {}, 'results': [], 'failures': [], 'summaries': {},
                'started_at_unix': time.time()}
    try:
        with open(args.manifest) as stream:
            manifest = json.load(stream)
        reference_root = resolve_reference_root(manifest, args.reference_root)
        artifact['reference_root'] = reference_root
        source_claim = manifest.get('reference_source', {})
        json.dumps(source_claim, allow_nan=False)
        artifact['reference_source'] = source_claim
        raw_thresholds = manifest.get('thresholds', {})
        artifact['thresholds'] = ({k: v if finite_number(v) or isinstance(v, str) else repr(v)
                                   for k, v in raw_thresholds.items()}
                                  if isinstance(raw_thresholds, dict) else repr(raw_thresholds))
        from reference_runner import Reference
        from app.models.wbia_orientation import WbiaOrientationModel
        artifact['environment'] = environment()
        rows, failures, summaries, checkpoints = evaluate(
            manifest, args.fixtures, reference_root, Reference, WbiaOrientationModel)
        artifact.update(results=rows, failures=failures, summaries=summaries, checkpoints=checkpoints)
    except Exception as exc:
        artifact['failures'].append(f'gate setup failed: {exc}')
    artifact['elapsed_seconds'] = time.monotonic() - started
    write_artifact(args.artifact, artifact)
    print(f'artifact -> {args.artifact}')
    if artifact['failures']:
        print('\n*** GATE FAILED — DO NOT DEPLOY ***')
        for failure in artifact['failures']:
            print(f'  {failure}')
        return 1
    print('\nGATE PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())

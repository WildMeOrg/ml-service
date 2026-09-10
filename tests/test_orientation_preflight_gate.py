import json
import math
import sys
import types

import numpy as np
import pytest
from PIL import Image

from scripts.preflight import reference_runner as runner
from scripts.preflight import run_gate as gate


class Reference:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, data, bbox):
        return {'theta': bbox[0] / 10, 'coords_normalized': [0.25] * 5,
                'effective_bbox': list(map(int, bbox))}


class Port:
    calls = []
    mutate = staticmethod(lambda rows: rows)

    def load(self, **kwargs):
        pass

    def predict_batch(self, data, bboxes):
        self.calls.append(bboxes)
        return self.mutate([Reference().predict(data, bbox) for bbox in bboxes])


@pytest.fixture
def case(tmp_path, monkeypatch):
    Image.new('RGB', (20, 10), (73, 12, 24)).save(tmp_path / 'image.png')
    weights = tmp_path / 'weights.pth'
    weights.write_bytes(b'fake')
    manifest = {
        'thresholds': {'theta_circular_max_rad': 1e-5, 'theta_circular_mean_rad': 1e-6,
                       'coords_elementwise_max': 1e-6, 'coords_elementwise_mean': 1e-7,
                       'effective_bbox': 'exact', 'predict_batch': 'exact count and order'},
        'checkpoints': [{'model_id': 'test', 'path': str(weights)}],
        'fixtures': [{'file': 'image.png', 'bbox': [0, 0, 8, 8], 'stratum': 'rgb'}],
        'strata': {'rgb': {'min_samples': 1}}}
    monkeypatch.setattr(Port, 'calls', [])
    monkeypatch.setattr(Port, 'mutate', staticmethod(lambda rows: rows))
    return manifest, tmp_path


def evaluate(case, port=Port, reference=Reference):
    manifest, path = case
    return gate.evaluate(manifest, str(path), '/reference', reference, port)


def mutation(monkeypatch, fn):
    monkeypatch.setattr(Port, 'mutate', staticmethod(fn))


def test_matching_outputs_pass(case):
    rows, failures, summaries, checkpoints = evaluate(case)
    assert failures == []
    assert len(rows) == 1
    assert all(summaries['test'][key] == 0 for key in gate.ERROR_LIMITS)
    assert checkpoints[0]['sha256'] == gate.sha256(case[1] / 'weights.pth')


@pytest.mark.parametrize('metric', ['theta_circular_mean_rad', 'coords_elementwise_mean'])
def test_mean_violation_below_maximum_fails(case, monkeypatch, metric):
    def change(rows):
        if metric.startswith('theta'):
            rows[0]['theta'] += 5e-6
        else:
            rows[0]['coords_normalized'] = [v + 5e-7 for v in rows[0]['coords_normalized']]
        return rows
    mutation(monkeypatch, change)
    failures = evaluate(case)[1]
    assert any(metric in f for f in failures)
    assert not any('_max' in f for f in failures)


@pytest.mark.parametrize('kind,message', [('missing', 'exactly 1 rows'), ('extra', 'exactly 1 rows'),
    ('bbox', 'effective_bbox mismatch'), ('nan_theta', 'finite theta'), ('inf_theta', 'finite theta'),
    ('short_coords', 'five finite coordinates'), ('nan_coords', 'five finite coordinates'),
    ('none', 'finite theta')])
def test_malformed_outputs_fail(case, monkeypatch, kind, message):
    def change(rows):
        if kind == 'missing': return []
        if kind == 'extra': return rows * 2
        if kind == 'none': return [None]
        if kind == 'bbox': rows[0]['effective_bbox'] = [1, 0, 8, 8]
        if kind == 'nan_theta': rows[0]['theta'] = float('nan')
        if kind == 'inf_theta': rows[0]['theta'] = float('inf')
        if kind == 'short_coords': rows[0]['coords_normalized'] = [0.25] * 4
        if kind == 'nan_coords': rows[0]['coords_normalized'][4] = float('nan')
        return rows
    mutation(monkeypatch, change)
    rows, failures, _, _ = evaluate(case)
    assert rows == []
    assert any(message in f for f in failures)


def batch(case):
    manifest, _ = case
    manifest['fixtures'][0].pop('bbox')
    manifest['fixtures'][0].update(bboxes=[[0, 0, 8, 8], [10, 0, 8, 8]], stratum='multi_detection')
    manifest['strata'] = {'multi_detection': {'min_samples': 1}}


def test_multi_detection_calls_one_real_batch(case):
    batch(case)
    rows, failures, summaries, _ = evaluate(case)
    assert failures == []
    assert Port.calls == [[[0, 0, 8, 8], [10, 0, 8, 8]]]
    assert [r['batch_index'] for r in rows] == [0, 1]
    assert summaries['test']['coverage']['multi_detection'] == 1


@pytest.mark.parametrize('predictions_only', [False, True])
def test_swapped_batch_fails_even_when_bbox_order_is_correct(case, monkeypatch, predictions_only):
    batch(case)
    def change(rows):
        if predictions_only:
            rows[0]['theta'], rows[1]['theta'] = rows[1]['theta'], rows[0]['theta']
            return rows
        return rows[::-1]
    mutation(monkeypatch, change)
    failures = evaluate(case)[1]
    assert any(('theta_circular_max' if predictions_only else 'effective_bbox mismatch') in f for f in failures)


def test_single_crop_cannot_claim_multi_detection(case):
    case[0]['fixtures'][0]['stratum'] = 'multi_detection'
    case[0]['strata'] = {'multi_detection': {'min_samples': 1}}
    assert any('two distinct' in f for f in evaluate(case)[1])


def test_indistinguishable_predictions_do_not_certify_batch_order(case):
    batch(case)
    class Same(Reference):
        def predict(self, data, bbox):
            result = super().predict(data, bbox)
            result['theta'] = 0
            return result
    assert any('cannot distinguish row order' in f for f in evaluate(case, reference=Same)[1])


def test_means_are_checked_per_checkpoint(case):
    manifest, _ = case
    manifest['checkpoints'].append({**manifest['checkpoints'][0], 'model_id': 'bad'})
    class PerModel(Port):
        def load(self, model_id, **kwargs): self.model_id = model_id
        def predict_batch(self, data, bboxes):
            rows = super().predict_batch(data, bboxes)
            if self.model_id == 'bad': rows[0]['theta'] += 1.5e-6
            return rows
    assert any('bad: theta_circular_mean_rad' in f for f in evaluate(case, port=PerModel)[1])


def test_coverage_is_per_checkpoint_and_deduplicated(case):
    manifest, _ = case
    manifest['checkpoints'].append({**manifest['checkpoints'][0], 'model_id': 'other'})
    manifest['fixtures'] *= 2
    manifest['strata']['rgb']['min_samples'] = 2
    assert sum('1 samples < required 2' in f for f in evaluate(case)[1]) == 2


@pytest.mark.parametrize('key', ['fixtures', 'checkpoints', 'strata'])
def test_empty_manifest_sections_fail(case, key):
    case[0][key] = {} if key == 'strata' else []
    assert any(key + ' must be nonempty' in f for f in evaluate(case)[1])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, True, '0.1'])
def test_invalid_numeric_thresholds_fail(case, value):
    case[0]['thresholds']['theta_circular_mean_rad'] = value
    assert any('must be finite and nonnegative' in f for f in evaluate(case)[1])


def test_exact_thresholds_pass_and_coord_mean_uses_all_five_components(case, monkeypatch):
    case[0]['thresholds'].update(theta_circular_max_rad=0.125, theta_circular_mean_rad=0.125,
                                 coords_elementwise_max=0.125, coords_elementwise_mean=0.025)
    def change(rows):
        rows[0]['theta'] = 0.125
        rows[0]['coords_normalized'][0] += 0.125
        return rows
    mutation(monkeypatch, change)
    _, failures, summaries, _ = evaluate(case)
    assert failures == []
    assert summaries['test']['coords_elementwise_mean'] == 0.025


@pytest.mark.parametrize('bbox', [[1, 1, 3, 2], [-2, 0, 20, 4], [-2, 0, 3, 4],
                                  [99, 0, 2, 2], [-0.5, 0.9, 3.8, 2.9], [7, 0, 3, 4]])
def test_reference_crop_matches_numpy_index_oracle(bbox):
    image = np.arange(32).reshape(4, 8)
    x, y, w, h = map(int, bbox)
    expected = image[y:y+h, x:x+w]
    if not expected.size: expected = image
    origin = int(expected[0, 0])
    crop, effective = runner.reference_crop(image, bbox)
    np.testing.assert_array_equal(crop, expected)
    assert effective == [origin % 8, origin // 8, expected.shape[1], expected.shape[0]]


def test_setup_failure_writes_artifact(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('invalid JSON')
    artifact = tmp_path / 'artifact.json'
    assert gate.main(['--manifest', str(manifest), '--fixtures', str(tmp_path), '--artifact', str(artifact)]) == 1
    assert 'gate setup failed' in json.loads(artifact.read_text())['failures'][0]


@pytest.mark.parametrize('failure', ['load', 'inference', 'nan_threshold'])
def test_failures_still_write_valid_artifact(case, monkeypatch, failure):
    manifest, path = case
    class Broken(Port):
        def load(self, **kwargs):
            if failure == 'load': raise RuntimeError('load broken')
        def predict_batch(self, *args): raise RuntimeError('inference broken')
    if failure == 'nan_threshold': manifest['thresholds']['theta_circular_max_rad'] = float('nan')
    monkeypatch.setitem(sys.modules, 'reference_runner', types.SimpleNamespace(Reference=Reference))
    monkeypatch.setitem(sys.modules, 'app.models.wbia_orientation', types.SimpleNamespace(WbiaOrientationModel=Broken))
    monkeypatch.setattr(gate, 'environment', lambda: {})
    (path / 'manifest.json').write_text(json.dumps(manifest))
    assert gate.main(['--manifest', str(path / 'manifest.json'), '--fixtures', str(path),
                      '--artifact', str(path / 'artifact.json')]) == 1
    artifact = json.loads((path / 'artifact.json').read_text(), parse_constant=lambda s: pytest.fail(s))
    assert artifact['failures']


def test_atomic_artifact_preserves_previous_json_on_serialization_error(tmp_path):
    path = tmp_path / 'artifact.json'
    path.write_text('{"previous": true}')
    with pytest.raises(ValueError): gate.write_artifact(path, {'error': float('nan')})
    assert json.loads(path.read_text()) == {'previous': True}
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('kind', ['theta', 'grayscale'])
def test_gate_detects_mutations_in_actual_port_without_real_weights(case, monkeypatch, kind):
    import torch
    from app.models import wbia_orientation as model
    class Backbone(torch.nn.Module):
        def forward(self, x):
            return torch.logit(torch.tensor([[0.5, 0.5, 0.75, 0.5, 0.125]])).repeat(len(x), 1)
    def load(self, **kwargs):
        self.model = Backbone()
        self.model_id = 'test'
        self.device = 'cpu'
        self.hflip = self.vflip = False
    monkeypatch.setattr(model.WbiaOrientationModel, 'load', load)
    class Oracle(Reference):
        def predict(self, data, bbox):
            return {'theta': math.pi / 2, 'coords_normalized': [0.5, 0.5, 0.75, 0.5, 0.125], 'effective_bbox': bbox}
    if kind == 'grayscale':
        Image.new('L', (20, 10), 73).save(case[1] / 'image.png')
        case[0]['fixtures'][0]['stratum'] = 'canonicalization_wrapper'
        case[0]['strata'] = {'canonicalization_wrapper': {'min_samples': 1}}
    assert evaluate(case, model.WbiaOrientationModel, Oracle)[1] == []
    if kind == 'theta':
        original = model.compute_theta
        monkeypatch.setattr(model, 'compute_theta', lambda c: original(c) - math.pi / 2)
    else:
        monkeypatch.setattr(model, '_canonicalize_rgb', lambda image: image[:, :, None])
    assert evaluate(case, model.WbiaOrientationModel, Oracle)[1]

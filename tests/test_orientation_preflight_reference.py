import subprocess
import sys
from pathlib import Path

import pytest

from scripts.preflight import reference_runner as runner
from scripts.preflight.run_gate import resolve_reference_root


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

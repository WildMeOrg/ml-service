"""Tests for ${MODEL_BASE} substitution in the model config.

The same config file has to work whether the weights sit on a bind mount
(/datasets on the VM), a provider volume (/runpod-volume/models), or behind an
https:// object-store prefix.

Substitution runs on parsed values in known weight-path fields only. Expanding
the raw file text instead would let a prefix containing a quote, backslash or
newline break the JSON, would rewrite an unrelated '$' in a model id, and on
Windows would expand %VAR%. Those are the cases most of this file guards.
"""
import json
import os

import pytest

from app.utils.config_loader import DEFAULT_MODEL_BASE, load_model_config


@pytest.fixture
def config_file(tmp_path):
    def write(models):
        path = tmp_path / "model_config.json"
        path.write_text(json.dumps({"models": models}))
        return str(path)
    return write


def test_defaults_to_datasets_when_model_base_unset(config_file, monkeypatch):
    """Existing deployments set nothing; they must keep resolving to /datasets."""
    monkeypatch.delenv("MODEL_BASE", raising=False)
    path = config_file([{"model_id": "m", "model_path": "${MODEL_BASE}/detect.pt"}])

    config = load_model_config(path)

    assert config["models"][0]["model_path"] == "/datasets/detect.pt", \
        "unset MODEL_BASE must fall back to the bind-mount path the VM already uses"
    assert DEFAULT_MODEL_BASE == "/datasets"


def test_empty_model_base_falls_back_to_default(config_file, monkeypatch):
    """An env file with 'MODEL_BASE=' must not resolve weights to bare /detect.pt."""
    monkeypatch.setenv("MODEL_BASE", "")
    path = config_file([{"model_id": "m", "model_path": "${MODEL_BASE}/detect.pt"}])

    assert load_model_config(path)["models"][0]["model_path"] == "/datasets/detect.pt"


def test_expands_object_store_prefix(config_file, monkeypatch):
    monkeypatch.setenv("MODEL_BASE", "https://storage.googleapis.com/bucket/models")
    path = config_file([
        {"model_id": "a", "model_path": "${MODEL_BASE}/detect.pt"},
        {"model_id": "b", "checkpoint_path": "${MODEL_BASE}/miew.bin"},
    ])

    config = load_model_config(path)

    assert config["models"][0]["model_path"] == \
        "https://storage.googleapis.com/bucket/models/detect.pt"
    assert config["models"][1]["checkpoint_path"] == \
        "https://storage.googleapis.com/bucket/models/miew.bin", \
        "checkpoint_path must be substituted too, not just model_path"


def test_expands_provider_volume_prefix(config_file, monkeypatch):
    monkeypatch.setenv("MODEL_BASE", "/runpod-volume/models")
    path = config_file([{"model_id": "m", "checkpoint_path": "${MODEL_BASE}/miew.bin"}])

    assert load_model_config(path)["models"][0]["checkpoint_path"] == \
        "/runpod-volume/models/miew.bin"


def test_trailing_slash_does_not_double_up(config_file, monkeypatch):
    monkeypatch.setenv("MODEL_BASE", "https://example.invalid/models/")
    path = config_file([{"model_id": "m", "model_path": "${MODEL_BASE}/detect.pt"}])

    assert load_model_config(path)["models"][0]["model_path"] == \
        "https://example.invalid/models/detect.pt", \
        "a trailing slash on the prefix must not produce a '//' in the URL"


def test_literal_paths_pass_through_untouched(config_file, monkeypatch):
    """A registry with literal paths ignores MODEL_BASE; that is documented, not a bug."""
    monkeypatch.setenv("MODEL_BASE", "https://example.invalid/models")
    path = config_file([{"model_id": "m", "model_path": "/datasets/detect.pt"}])

    assert load_model_config(path)["models"][0]["model_path"] == "/datasets/detect.pt"


def test_other_environment_variables_are_not_expanded(config_file, monkeypatch):
    """Only ${MODEL_BASE} is substituted -- never arbitrary shell-style vars."""
    monkeypatch.setenv("MODEL_BASE", "/datasets")
    monkeypatch.setenv("HOME", "/home/someone")
    path = config_file([{"model_id": "m", "model_path": "$HOME/${OTHER}/detect.pt"}])

    assert load_model_config(path)["models"][0]["model_path"] == \
        "$HOME/${OTHER}/detect.pt", \
        "an unrelated $VAR must survive verbatim"


def test_non_path_fields_are_never_touched(config_file, monkeypatch):
    """Model ids and labels can legitimately contain '$'."""
    monkeypatch.setenv("MODEL_BASE", "/datasets")
    path = config_file([{
        "model_id": "cost-$MODEL_BASE-v1",
        "note": "${MODEL_BASE} mentioned in prose",
        "model_path": "${MODEL_BASE}/detect.pt",
    }])

    model = load_model_config(path)["models"][0]

    assert model["model_id"] == "cost-$MODEL_BASE-v1"
    assert model["note"] == "${MODEL_BASE} mentioned in prose", \
        "substitution must be confined to weight-path fields"
    assert model["model_path"] == "/datasets/detect.pt"


def test_prefix_with_json_metacharacters_cannot_break_parsing(config_file, monkeypatch):
    """The whole reason substitution happens after json.load, not before."""
    hostile = '/weights"/x\\y'
    monkeypatch.setenv("MODEL_BASE", hostile)
    path = config_file([{"model_id": "m", "model_path": "${MODEL_BASE}/detect.pt"}])

    config = load_model_config(path)  # would raise JSONDecodeError if text-expanded

    assert config["models"][0]["model_path"] == hostile + "/detect.pt"


def test_process_environment_is_not_mutated(config_file, monkeypatch):
    """Loading a config must not leak a MODEL_BASE default into child processes."""
    monkeypatch.delenv("MODEL_BASE", raising=False)
    path = config_file([{"model_id": "m", "model_path": "${MODEL_BASE}/detect.pt"}])

    load_model_config(path)

    assert "MODEL_BASE" not in os.environ, \
        "the default must stay local; setting it in os.environ makes it inherited and sticky"


def test_list_valued_checkpoint_paths_are_substituted(config_file, monkeypatch):
    """DenseNet ensembles configure a list, not a single path."""
    monkeypatch.setenv("MODEL_BASE", "/vol/models")
    path = config_file([{
        "model_id": "ensemble",
        "checkpoint_paths": ["${MODEL_BASE}/a.pt", "${MODEL_BASE}/b.pt"],
    }])

    assert load_model_config(path)["models"][0]["checkpoint_paths"] == \
        ["/vol/models/a.pt", "/vol/models/b.pt"]


def test_lightnet_config_and_weight_paths_are_substituted(config_file, monkeypatch):
    """LightNet names its two locations config_path and weight_path."""
    monkeypatch.setenv("MODEL_BASE", "/vol/models")
    path = config_file([{
        "model_id": "ln",
        "config_path": "${MODEL_BASE}/cfg.py",
        "weight_path": "${MODEL_BASE}/w.weights",
    }])

    model = load_model_config(path)["models"][0]

    assert model["config_path"] == "/vol/models/cfg.py"
    assert model["weight_path"] == "/vol/models/w.weights"


def test_nested_role_checkpoints_are_substituted(config_file, monkeypatch):
    """The wild dog cascade nests a checkpoint under each role object."""
    monkeypatch.setenv("MODEL_BASE", "/vol/models")
    path = config_file([{
        "model_id": "cascade",
        "model_type": "densenet-wilddog-cascade",
        "router": {"checkpoint_path": "${MODEL_BASE}/router.pt"},
        "coat": {"checkpoint_paths": ["${MODEL_BASE}/coat1.pt", "${MODEL_BASE}/coat2.pt"]},
        "viewpoint": {"checkpoint_path": "${MODEL_BASE}/vp.pt", "img_size": 224},
    }])

    model = load_model_config(path)["models"][0]

    assert model["router"]["checkpoint_path"] == "/vol/models/router.pt"
    assert model["coat"]["checkpoint_paths"] == \
        ["/vol/models/coat1.pt", "/vol/models/coat2.pt"]
    assert model["viewpoint"]["checkpoint_path"] == "/vol/models/vp.pt"
    assert model["viewpoint"]["img_size"] == 224, "non-path keys survive the recursion"


def test_unset_model_base_preserves_legacy_resolution_exactly(monkeypatch):
    """The only backward-compatibility claim worth making: unset means unchanged."""
    monkeypatch.delenv("MODEL_BASE", raising=False)

    config = load_model_config("app/model_config.json")
    resolved = {m[f] for m in config["models"]
                for f in ("model_path", "checkpoint_path") if f in m}

    assert resolved == {
        "/datasets/detect.yolov11.msv3.pt",
        "/datasets/miew_id.msv4_1_main.bin",
        "/datasets/miewid_trout.bin",
        "/datasets/vplabeler-msv3.pt",
    }, "the shipped registry must resolve to the exact paths it used before"


def test_setting_model_base_now_relocates_the_shipped_registry(monkeypatch):
    """The flip side, stated plainly: a set value takes effect from now on."""
    monkeypatch.setenv("MODEL_BASE", "https://example.invalid/weights")

    config = load_model_config("app/model_config.json")
    resolved = [m[f] for m in config["models"]
                for f in ("model_path", "checkpoint_path") if f in m]

    assert all(p.startswith("https://example.invalid/weights/") for p in resolved), \
        "MODEL_BASE is new, so nothing set it before; from here it moves these paths"


def test_committed_default_config_resolves_to_the_legacy_paths(monkeypatch):
    """Guard the real file: the shipped registry must resolve exactly as before."""
    monkeypatch.delenv("MODEL_BASE", raising=False)

    config = load_model_config("app/model_config.json")

    assert config["models"], "the committed config must still load"
    resolved = [m[f] for m in config["models"]
                for f in ("model_path", "checkpoint_path") if f in m]
    assert resolved, "the committed config must declare some weight paths"
    for path in resolved:
        assert path.startswith("/datasets/"), \
            f"default resolution changed for {path!r}; existing installs would break"
        assert "${" not in path

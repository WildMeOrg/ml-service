"""Tests for ${MODEL_BASE} expansion in the model config.

The same config file has to work whether the weights sit on a bind mount
(/datasets on the VM), a provider volume (/runpod-volume/models), or behind an
https:// object-store prefix. Expanding the prefix from the environment at load
time is what makes one file portable across all three.
"""
import json

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
        "checkpoint_path must expand too, not just model_path"


def test_expands_provider_volume_prefix(config_file, monkeypatch):
    monkeypatch.setenv("MODEL_BASE", "/runpod-volume/models")
    path = config_file([{"model_id": "m", "checkpoint_path": "${MODEL_BASE}/miew.bin"}])

    assert load_model_config(path)["models"][0]["checkpoint_path"] == \
        "/runpod-volume/models/miew.bin"


def test_literal_paths_pass_through_untouched(config_file, monkeypatch):
    """The committed config uses literal /datasets paths; expansion is a no-op."""
    monkeypatch.setenv("MODEL_BASE", "https://example.invalid/models")
    path = config_file([{"model_id": "m", "model_path": "/datasets/detect.pt"}])

    assert load_model_config(path)["models"][0]["model_path"] == "/datasets/detect.pt", \
        "a config with no ${MODEL_BASE} reference must be unaffected by the env var"


def test_committed_default_config_still_parses(monkeypatch):
    """Guard the real file: expansion must not break the shipped registry."""
    monkeypatch.delenv("MODEL_BASE", raising=False)

    config = load_model_config("app/model_config.json")

    assert config["models"], "the committed config must still load"
    for model in config["models"]:
        for key in ("model_path", "checkpoint_path"):
            if key in model:
                assert "${" not in model[key], \
                    f"{model['model_id']}.{key} left an unexpanded reference"

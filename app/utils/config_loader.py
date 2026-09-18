"""Model-config loading.

``${VAR}`` references in the config body are expanded from the environment
before the JSON is parsed, so one config file points at any model store:
``/datasets`` (the VM bind mount), a provider volume such as
``/runpod-volume/models``, or an ``https://`` object-store prefix. URL values
are fetched and cached at load time by ``checkpoint_utils.get_checkpoint_path``.

``MODEL_BASE`` defaults to ``/datasets``, which is the path the committed
config and the production compose file already use — so expansion is a no-op
for existing deployments.
"""
import json
import os

DEFAULT_MODEL_BASE = "/datasets"


def load_model_config(config_path: str) -> dict:
    """Read a model config, expanding ${VAR} references from the environment."""
    os.environ.setdefault("MODEL_BASE", DEFAULT_MODEL_BASE)
    with open(config_path, "r") as f:
        return json.loads(os.path.expandvars(f.read()))

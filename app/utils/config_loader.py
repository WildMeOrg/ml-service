"""Model-config loading.

``${MODEL_BASE}`` at the start of a weight path in the model config is replaced
with a per-environment prefix, so one config file serves any model store:
``/datasets`` (the VM bind mount), a provider volume such as
``/runpod-volume/models``, or an ``https://`` object-store prefix that
``checkpoint_utils`` fetches and caches at load time.

The substitution is deliberately narrow — it runs on parsed values, only in the
known weight-path fields, and only for the single ``MODEL_BASE`` token. Doing it
on the raw file text instead would let an expansion inject a quote, backslash or
newline into the JSON, rewrite an unrelated ``$`` in a model id, or (on Windows)
expand ``%VAR%``. ``MODEL_BASE`` defaults to ``/datasets``, which is what the
committed config and the production compose file already resolve to, so this is
a no-op for existing deployments.
"""
import json
import os

DEFAULT_MODEL_BASE = "/datasets"

#: Config keys holding a weight location. Anything else (model ids, labels) is
#: left exactly as written.
PATH_FIELDS = ("model_path", "checkpoint_path")

_TOKEN = "${MODEL_BASE}"


def _substitute(value, model_base):
    if not isinstance(value, str) or _TOKEN not in value:
        return value
    return value.replace(_TOKEN, model_base)


def load_model_config(config_path: str) -> dict:
    """Read a model config, resolving ${MODEL_BASE} in weight-path fields."""
    model_base = os.environ.get("MODEL_BASE") or DEFAULT_MODEL_BASE
    model_base = model_base.rstrip("/")

    with open(config_path, "r") as f:
        config = json.load(f)

    for model in config.get("models", []):
        for field in PATH_FIELDS:
            if field in model:
                model[field] = _substitute(model[field], model_base)
    return config

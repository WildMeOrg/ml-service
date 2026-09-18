"""Model-config loading.

``${MODEL_BASE}`` in a weight location in the model config is replaced with a
per-environment prefix, so one config file serves any model store: ``/datasets``
(the VM bind mount), a provider volume such as ``/runpod-volume/models``, or an
``https://`` object-store prefix that ``checkpoint_utils`` fetches and caches at
load time.

The substitution is deliberately narrow. It runs on parsed values, only under
keys known to hold a weight location, and only for the single ``MODEL_BASE``
token. Doing it on the raw file text instead would let a prefix inject a quote,
backslash or newline into the JSON, rewrite an unrelated ``$`` in a model id, or
(on Windows) expand ``%VAR%``.

Traversal is recursive because weight locations are not all top-level strings:
``checkpoint_paths`` is a list, and the wild dog cascade nests a
``checkpoint_path`` inside each of its ``router`` / ``coat`` / ``viewpoint``
role objects. Recursing on the key name rather than enumerating those roles
means a new nested model type is covered without touching this file.

``MODEL_BASE`` defaults to ``/datasets``, which is what the committed config
already resolved to, so leaving it unset changes nothing.
"""
import json
import os

DEFAULT_MODEL_BASE = "/datasets"

#: Keys whose values hold a weight location, as accepted by the model loaders:
#: model_path (yolo/base), checkpoint_path (miewid, megadetector, densenet,
#: efficientnet), checkpoint_paths (densenet ensembles), and config_path /
#: weight_path (lightnet). Anything else -- model ids, labels, thresholds -- is
#: left exactly as written.
PATH_FIELDS = frozenset({
    "model_path",
    "checkpoint_path",
    "checkpoint_paths",
    "config_path",
    "weight_path",
})

TOKEN = "${MODEL_BASE}"


def _substitute(value, model_base):
    """Replace the token in a weight location: a string, or a list of them."""
    if isinstance(value, str):
        return value.replace(TOKEN, model_base)
    if isinstance(value, list):
        return [_substitute(item, model_base) for item in value]
    return value


def _walk(node, model_base):
    """Recurse, substituting only under PATH_FIELDS keys."""
    if isinstance(node, dict):
        return {
            key: (_substitute(value, model_base) if key in PATH_FIELDS
                  else _walk(value, model_base))
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_walk(item, model_base) for item in node]
    return node


def load_model_config(config_path: str) -> dict:
    """Read a model config, resolving ${MODEL_BASE} in every weight location."""
    model_base = (os.environ.get("MODEL_BASE") or DEFAULT_MODEL_BASE).rstrip("/")

    with open(config_path, "r") as f:
        config = json.load(f)

    return _walk(config, model_base)

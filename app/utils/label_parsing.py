"""Shared parser for class labels of the form `<species>:<viewpoint>`.

Used by both DenseNetClassifierModel and EfficientNetModel so compound-
label parsing behavior is identical across model types.
"""
from typing import List, Optional, Tuple

DEFAULT_SENTINEL_PREFIXES = ["species"]


def parse_class_label(
    label: str,
    compound_labels: bool,
    sentinel_prefixes: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Parse a class label into (species, viewpoint).

    Args:
        label: the raw class label string.
        compound_labels: whether labels are expected to be `<species>:<viewpoint>`.
        sentinel_prefixes: list of placeholder prefixes whose appearance suppresses
            species emission. Defaults to ['species'] (matches the deployed
            efficientnet-classifier convention where `species:` is a literal
            namespace, not a real species name). Pass [] to disable.

    Returns:
        (species, viewpoint). species is None when:
          - compound_labels is False and the label has no colon, OR
          - compound_labels is True but the label has no colon, OR
          - the prefix matches a sentinel.
        viewpoint is always populated (the whole label when no colon, or the
        suffix after the first colon when colon-bearing).

    Raises:
        ValueError: if compound_labels is False but the label contains ':'.
            This catches a config mistake (operator forgot to set
            compound_labels: true) at the earliest opportunity.
    """
    if sentinel_prefixes is None:
        sentinel_prefixes = DEFAULT_SENTINEL_PREFIXES

    if ":" not in label:
        return (None, label)

    if not compound_labels:
        raise ValueError(
            f"Label {label!r} contains ':' but compound_labels=False. "
            f"Either set compound_labels: true in the model config, "
            f"or fix the checkpoint to use non-compound labels."
        )

    prefix, suffix = label.split(":", 1)
    if prefix in sentinel_prefixes:
        return (None, suffix)
    return (prefix, suffix)

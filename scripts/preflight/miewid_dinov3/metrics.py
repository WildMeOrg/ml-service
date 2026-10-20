"""Parity-gate metrics for the MiewID DINOv3 port. Pure numpy, no torch, so
the same code scores runs from either environment."""
import numpy as np


def router_bbox(bbox):
    """Exactly /extract's conversion (extract_router.py): int() truncates
    toward zero, so -213.7 -> -213, not -214."""
    return tuple(int(v) for v in bbox)


def _pair(got, ref):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    if got.shape != ref.shape:
        raise ValueError(f"shape mismatch: got {got.shape}, ref {ref.shape}")
    return got, ref


def max_abs(got, ref) -> float:
    got, ref = _pair(got, ref)
    return float(np.max(np.abs(got - ref)))


def mean_abs(got, ref) -> float:
    got, ref = _pair(got, ref)
    return float(np.mean(np.abs(got - ref)))


def rel_l2(got, ref) -> float:
    got, ref = _pair(got, ref)
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def cosine_distance(a, b) -> float:
    a, b = _pair(a, b)
    a, b = a.ravel(), b.ravel()
    return float(1.0 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _topk(emb, k):
    e = np.asarray(emb, np.float64)
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    sim = e @ e.T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1, kind="stable")[:, :k]


def topk_agreement(got, ref, k: int) -> float:
    """Fraction of queries whose ordered top-k neighbours match exactly."""
    got, _ = _pair(got, ref)
    if got.shape[0] <= k:
        raise ValueError(
            f"top-{k} needs more than {k} rows (self is excluded), got {got.shape[0]} rows")
    return float(np.mean(np.all(_topk(got, k) == _topk(ref, k), axis=1)))


def check_stage(name, got, ref, limits) -> dict:
    result = {"stage": name, "max_abs": max_abs(got, ref), "rel_l2": rel_l2(got, ref)}
    result["pass"] = result["max_abs"] <= limits["max_abs"] and result["rel_l2"] <= limits["rel_l2"]
    return result


def check_chip(got, ref, limits) -> dict:
    """uint8 HWC chips. A size mismatch is a failure, not an error: it is a
    crop/rotation disagreement, which is exactly what this stage looks for."""
    got, ref = np.asarray(got), np.asarray(ref)
    if got.shape != ref.shape:
        return {"shape_equal": False, "got_shape": list(got.shape),
                "ref_shape": list(ref.shape), "pass": False}
    result = {"shape_equal": True, "max_abs": max_abs(got, ref), "mean_abs": mean_abs(got, ref)}
    result["pass"] = result["max_abs"] <= limits["max_abs"] and result["mean_abs"] <= limits["mean_abs"]
    return result


def _key(chip):
    return (str(chip["case"]), str(chip["species"]), str(chip["name"]))


def check_expected_chips(rows, expected) -> list:
    """Problems (empty = ok) if the reference rows are not EXACTLY the expected
    gallery, in order. Rejects missing, extra, duplicate and reordered rows."""
    problems = []
    if len(rows) != len(expected):
        problems.append(f"{len(rows)} reference rows, expected {len(expected)}")
    for i, (row, exp) in enumerate(zip(rows, expected)):
        if _key(row) != _key(exp):
            problems.append(f"row {i}: {_key(row)} != expected {_key(exp)}")
    return problems


def check_known_divergence(known, expected) -> list:
    """Every known_divergence entry must name a row of expected_chips."""
    keys = [_key(e) for e in expected]
    return [f"known_divergence entry {_key(k)} matches no expected chip"
            for k in known if _key(k) not in keys]


def select_rows(expected, known) -> dict:
    """arch: every row. gated: e2e rows that must pass. reported: known-divergence
    rows, computed and reported but never gated."""
    skip = {_key(k) for k in known}
    reported = [i for i, e in enumerate(expected) if _key(e) in skip]
    gated = [i for i, e in enumerate(expected) if _key(e) not in skip]
    return {"arch": list(range(len(expected))), "gated": gated, "reported": reported}


def cosine_gate(cos, limit) -> dict:
    """NaN-safe: a non-finite cosine fails the stage."""
    cos = np.asarray(cos, np.float64)
    worst = float(np.max(cos))
    return {"max": worst, "pass": bool(np.all(np.isfinite(cos)) and worst <= limit)}


def exit_code(passed, thresholds_status, allow_proposed) -> int:
    if not passed:
        return 1
    if thresholds_status != "agreed" and not allow_proposed:
        return 2
    return 0

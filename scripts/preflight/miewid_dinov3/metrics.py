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
    _pair(got, ref)
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

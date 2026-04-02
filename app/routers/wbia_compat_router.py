"""WBIA-compatible API endpoints for backward compatibility with Wildbook.

Mimics WBIA's async job queue pattern and response format so Wildbook can
call ml-service as a drop-in replacement for WBIA's detection/labeling/
orientation/assignment pipeline.

WBIA flow:
1. POST /api/engine/detect/cnn/ → returns jobid
2. GET  /api/engine/job/status/?jobid=X → {"jobstatus": "completed"}
3. GET  /api/engine/job/result/?jobid=X → {"json_result": {...}}

All responses are wrapped in: {"status": {"success": true, ...}, "response": <data>}
"""

import collections
import logging
import threading
import uuid as uuid_mod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WBIA Compatibility"])

# In-memory job store with bounded size (LRU eviction of oldest completed jobs).
_MAX_JOBS = 10000
_jobs: Dict[str, Dict[str, Any]] = collections.OrderedDict()
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=4)


def _set_job(jobid: str, status: str, result=None, error=None):
    """Thread-safe job state update. Sets result before status to avoid races."""
    with _jobs_lock:
        if jobid not in _jobs:
            _jobs[jobid] = {}
        job = _jobs[jobid]
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        job["status"] = status
        # Evict oldest completed jobs if over limit
        if len(_jobs) > _MAX_JOBS:
            to_remove = []
            for jid, jdata in _jobs.items():
                if jdata.get("status") in ("completed", "exception") and jid != jobid:
                    to_remove.append(jid)
                if len(_jobs) - len(to_remove) <= _MAX_JOBS:
                    break
            for jid in to_remove:
                del _jobs[jid]


def _get_job(jobid: str) -> Optional[Dict[str, Any]]:
    """Thread-safe job lookup."""
    with _jobs_lock:
        return _jobs.get(jobid, {}).copy() if jobid in _jobs else None


# ---------------------------------------------------------------------------
# WBIA response wrapper
# ---------------------------------------------------------------------------

def _wbia_response(data, success=True, code="", message=""):
    """Wrap data in WBIA's standard response envelope."""
    return {
        "status": {
            "success": success,
            "code": code,
            "message": message,
            "cache": -1,
        },
        "response": data,
    }


# ---------------------------------------------------------------------------
# Job management endpoints
# ---------------------------------------------------------------------------

@router.get("/api/engine/job/status/")
@router.post("/api/engine/job/status/")
async def get_job_status(jobid: str = None):
    """Return the status of a job (matches WBIA /api/engine/job/status/)."""
    job = _get_job(jobid)
    if job is None:
        return _wbia_response({"jobstatus": "unknown"})
    return _wbia_response({"jobstatus": job.get("status", "unknown")})


@router.get("/api/engine/job/result/")
@router.post("/api/engine/job/result/")
async def get_job_result(jobid: str = None):
    """Return the result of a completed job (matches WBIA /api/engine/job/result/)."""
    job = _get_job(jobid)
    if job is None:
        return _wbia_response({"json_result": None}, success=False, message="Job not found")
    if job.get("status") == "exception":
        return _wbia_response(
            {"json_result": None},
            success=False,
            message=str(job.get("error", "Unknown error")),
        )
    return _wbia_response({"json_result": job.get("result")})


@router.get("/api/engine/job/")
@router.post("/api/engine/job/")
async def get_job_list():
    """Return list of job IDs (matches WBIA /api/engine/job/)."""
    with _jobs_lock:
        return _wbia_response(list(_jobs.keys()))


# ---------------------------------------------------------------------------
# Detection + pipeline endpoint
# ---------------------------------------------------------------------------

class WbiaDetectRequest(BaseModel):
    """Request matching WBIA's /api/engine/detect/cnn/ parameters."""
    image_uuid_list: List[str]
    model_tag: Optional[str] = None
    # Labeler (classification) params
    labeler_algo: Optional[str] = None
    labeler_model_tag: Optional[str] = None
    viewpoint_model_tag: Optional[str] = None
    use_labeler_species: bool = False
    # Assigner params
    assigner_algo: Optional[str] = None
    assigner_model_tag: Optional[str] = None
    # Detection params
    sensitivity: Optional[float] = None
    nms_thresh: Optional[float] = None
    nms_aware: Optional[str] = None
    # Callback (accepted but not implemented — Wildbook polls instead)
    callback_url: Optional[str] = None
    callback_method: Optional[str] = None
    callback_detailed: bool = False


@router.post("/api/engine/detect/cnn/yolo/")
@router.post("/api/engine/detect/cnn/lightnet/")
@router.post("/api/engine/detect/cnn/")
async def start_detect(request: Request, body: WbiaDetectRequest, background_tasks: BackgroundTasks):
    """WBIA-compatible detection endpoint.

    Accepts image URIs, runs detection (+ optional labeling/assignment),
    returns a jobid for async polling.
    """
    jobid = str(uuid_mod.uuid4())
    _set_job(jobid, "received")

    handler = request.app.state.model_handler

    background_tasks.add_task(_run_pipeline_async, jobid, body, handler)

    return _wbia_response(jobid)


async def _run_pipeline_async(jobid: str, body: WbiaDetectRequest, handler):
    """Wrapper to run pipeline in thread executor."""
    import asyncio
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, _run_pipeline_sync, jobid, body, handler)


def _run_pipeline_sync(jobid: str, body: WbiaDetectRequest, handler):
    """Run the full detect → label → assign pipeline synchronously in a thread."""
    try:
        _set_job(jobid, "working")

        image_uris = body.image_uuid_list
        model_tag = body.model_tag
        labeler_model_tag = body.labeler_model_tag or body.viewpoint_model_tag
        use_labeler_species = body.use_labeler_species
        assigner_algo = body.assigner_algo
        sensitivity = body.sensitivity

        results_list = []
        image_uuid_list = []

        for image_uri in image_uris:
            image_uuid_list.append(image_uri)

            # Load image once for all pipeline steps
            image_bytes = _load_image(image_uri)
            if image_bytes is None:
                results_list.append([])
                continue

            # --- Step 1: Detection ---
            detections = _run_detection(handler, model_tag, image_bytes, sensitivity)

            if not detections:
                results_list.append([])
                continue

            # --- Step 2: Labeling (classification) ---
            if labeler_model_tag:
                _run_labeling(handler, labeler_model_tag, image_bytes, detections, use_labeler_species)

            # --- Step 3: Assignment ---
            if assigner_algo:
                annot_results = _run_assignment(handler, detections, image_bytes)
            else:
                annot_results = [_strip_internal(d) for d in detections]

            results_list.append(annot_results)

        has_assignments = assigner_algo is not None
        result = {
            "image_uuid_list": image_uuid_list,
            "results_list": results_list,
            "score_list": [0.0] * len(image_uris),
            "has_assignments": has_assignments,
        }

        # Set result BEFORE status to avoid race where poller sees
        # "completed" but result is still None.
        _set_job(jobid, "completed", result=result)

    except Exception as e:
        logger.error(f"Job {jobid} failed: {e}", exc_info=True)
        _set_job(jobid, "exception", error=str(e))


def _run_detection(handler, model_tag, image_bytes, sensitivity):
    """Run detection model and return list of annotation dicts."""
    if not model_tag:
        return []

    model = handler.get_model(model_tag)
    if not model:
        logger.warning(f"Detection model '{model_tag}' not found")
        return []

    result = model.predict(image_bytes=image_bytes)

    bboxes = result.get("bboxes", [])
    scores = result.get("scores", [])
    class_names = result.get("class_names", [])
    thetas = result.get("thetas", None)
    # Handle missing or empty thetas
    if not thetas:
        thetas = [0.0] * len(bboxes)

    # Apply sensitivity filter
    if sensitivity is not None:
        filtered = [
            (b, s, c, t) for b, s, c, t in zip(bboxes, scores, class_names, thetas)
            if s >= sensitivity
        ]
        if filtered:
            bboxes, scores, class_names, thetas = map(list, zip(*filtered))
        else:
            return []

    detections = []
    for i, (bbox, score, cls, theta) in enumerate(zip(bboxes, scores, class_names, thetas)):
        det = {
            "id": i + 1,
            "uuid": str(uuid_mod.uuid4()),
            "xtl": int(round(bbox[0])),
            "ytl": int(round(bbox[1])),
            "left": int(round(bbox[0])),
            "top": int(round(bbox[1])),
            "width": int(round(bbox[2])),
            "height": int(round(bbox[3])),
            "theta": round(float(theta), 4),
            "confidence": round(float(score), 4),
            "class": cls,
            "species": cls,
            "viewpoint": None,
            "quality": None,
            "multiple": False,
            "interest": False,
            # Internal field for pipeline (stripped before returning to client)
            "_bbox": bbox,
        }
        detections.append(det)

    return detections


def _run_labeling(handler, labeler_model_tag, image_bytes, detections, use_labeler_species):
    """Run classification on each detection bbox, updating detections in place."""
    model = handler.get_model(labeler_model_tag)
    if not model:
        logger.warning(f"Labeler model '{labeler_model_tag}' not found")
        return

    for det in detections:
        try:
            bbox = [det["xtl"], det["ytl"], det["width"], det["height"]]
            result = model.predict(
                image_bytes=image_bytes,
                bbox=bbox,
                theta=det["theta"],
            )

            predictions = result.get("predictions", [])
            if predictions:
                top = predictions[0]
                # Handle compound labels (species:viewpoint)
                if "viewpoint" in top:
                    det["viewpoint"] = top["viewpoint"]
                elif ":" in top.get("label", ""):
                    parts = top["label"].split(":", 1)
                    det["viewpoint"] = parts[1] if len(parts) > 1 else None
                else:
                    det["viewpoint"] = top.get("label")

                if use_labeler_species:
                    if "species" in top:
                        det["species"] = top["species"]
                        det["class"] = top["species"]
                    elif ":" in top.get("label", ""):
                        det["species"] = top["label"].split(":")[0]
                        det["class"] = det["species"]

        except Exception as e:
            logger.warning(f"Labeling failed for detection {det['id']}: {e}")


def _run_assignment(handler, detections, image_bytes):
    """Run assigner on detections, returning grouped annotation results.

    When assignments are made, returns lists of lists (assigned groups).
    Without assignments, returns flat annotation dicts wrapped in single-element lists.
    """
    # In WBIA, species with '+' suffix are parts (e.g., 'lion+head')
    parts = [d for d in detections if "+" in d.get("species", "")]
    bodies = [d for d in detections if "+" not in d.get("species", "")]

    if not parts or not bodies:
        return [[_strip_internal(d)] for d in detections]

    from app.models.assigner import compute_pair_features, make_assignments, AssignerHandler
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return [[_strip_internal(d)] for d in detections]
    img_h, img_w = img.shape[:2]

    species = bodies[0].get("species", "")
    if not species:
        logger.warning("No species on body detection, cannot run assigner")
        return [[_strip_internal(d)] for d in detections]

    if not hasattr(_run_assignment, "_assigner"):
        _run_assignment._assigner = AssignerHandler()
    assigner = _run_assignment._assigner

    try:
        clf = assigner.get_classifier(species)
        feature_type = assigner.get_feature_type(species)
    except Exception as e:
        logger.warning(f"Could not load assigner for species '{species}': {e}")
        return [[_strip_internal(d)] for d in detections]

    part_aids = []
    body_aids = []
    features_list = []

    for part in parts:
        for body in bodies:
            part_bbox = [part["xtl"], part["ytl"], part["width"], part["height"]]
            body_bbox = [body["xtl"], body["ytl"], body["width"], body["height"]]
            feats = compute_pair_features(
                part_bbox=part_bbox, part_theta=part["theta"],
                part_viewpoint=part.get("viewpoint"),
                body_bbox=body_bbox, body_theta=body["theta"],
                body_viewpoint=body.get("viewpoint"),
                image_width=img_w, image_height=img_h,
                feature_type=feature_type,
            )
            part_aids.append(part["id"])
            body_aids.append(body["id"])
            features_list.append(feats)

    scores = clf.predict_proba(features_list)[:, 1].tolist()

    assigned, unassigned_ids = make_assignments(
        part_aids=part_aids,
        body_aids=body_aids,
        scores=scores,
        cutoff_score=0.5,
    )

    # Build result: assigned pairs as [body, part], unassigned as [annot]
    det_by_id = {d["id"]: _strip_internal(d) for d in detections}
    results = []
    seen_ids = set()

    for pair in assigned:
        results.append([det_by_id[pair["body_aid"]], det_by_id[pair["part_aid"]]])
        seen_ids.add(pair["part_aid"])
        seen_ids.add(pair["body_aid"])

    for aid in unassigned_ids:
        if aid in det_by_id and aid not in seen_ids:
            results.append([det_by_id[aid]])
            seen_ids.add(aid)

    return results


def _strip_internal(det: dict) -> dict:
    """Remove internal fields (prefixed with _) from a detection dict."""
    return {k: v for k, v in det.items() if not k.startswith("_")}


def _load_image(image_uri: str) -> Optional[bytes]:
    """Load image bytes from a URI (URL or local file path)."""
    try:
        if image_uri.startswith(("http://", "https://")):
            resp = httpx.get(image_uri, timeout=30)
            resp.raise_for_status()
            return resp.content
        else:
            with open(image_uri, "rb") as f:
                return f.read()
    except Exception as e:
        logger.error(f"Failed to load image '{image_uri}': {e}")
        return None


# ---------------------------------------------------------------------------
# Labeler-only endpoint (stub — Wildbook uses the combined detect endpoint)
# ---------------------------------------------------------------------------

@router.post("/api/engine/labeler/cnn/")
async def start_labeler(request: Request, body: dict):
    """WBIA-compatible labeler endpoint (stub).

    This endpoint exists for URL compatibility but is not fully implemented.
    Wildbook's primary flow uses the combined /api/engine/detect/cnn/ endpoint
    with labeler_model_tag to run labeling as part of the detection pipeline.
    """
    jobid = str(uuid_mod.uuid4())
    _set_job(jobid, "completed", result={
        "status": "not_implemented",
        "message": "Use /api/engine/detect/cnn/ with labeler_model_tag instead",
    })
    return _wbia_response(jobid)

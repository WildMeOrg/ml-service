"""Tests for WBIA-compatible API endpoints.

Covers:
- Response envelope format (status wrapper)
- Job queue lifecycle (submit → poll → result)
- Detection-only pipeline
- Detection + labeling pipeline
- Multiple images / batch processing
- Edge cases (missing models, bad images, empty results)
- Thread safety of job store
- Internal field stripping
"""

import os
import time
import threading
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import wbia_compat_router
from app.routers.wbia_compat_router import (
    _wbia_response, _strip_internal, _run_detection, _run_labeling,
    _set_job, _get_job, _jobs, _jobs_lock,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_client():
    """Test client with real LightNet + DenseNet models loaded."""
    from app.models.model_handler import ModelHandler

    app = FastAPI()
    app.include_router(wbia_compat_router.router)

    handler = ModelHandler()
    handler.load_model(
        model_id="detect-hyaena",
        model_type="lightnet",
        device="cpu",
        config_path="/mnt/c/claude-skills/models/reference/lightnet/detect.lightnet.hyaena.v0.py",
        weight_path="/mnt/c/claude-skills/models/reference/lightnet/detect.lightnet.hyaena.v0.weights",
    )
    handler.load_model(
        model_id="labeler-hyaena",
        model_type="densenet-orientation",
        device="cpu",
        checkpoint_path="/mnt/c/claude-skills/models/reference/wbia/labeler.hyaena.v0/labeler.hyaena/labeler.0.weights",
    )
    app.state.model_handler = handler

    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_client():
    """Test client with mock models for fast unit tests."""
    app = FastAPI()
    app.include_router(wbia_compat_router.router)

    handler = MagicMock()

    # Mock detection model
    detect_model = MagicMock()
    detect_model.predict.return_value = {
        "bboxes": [[10, 20, 100, 80], [200, 150, 50, 60]],
        "scores": [0.95, 0.70],
        "class_names": ["hyaena", "hyaena"],
        "thetas": [0.0, 0.1],
    }

    # Mock labeler model
    labeler_model = MagicMock()
    labeler_model.predict.return_value = {
        "predictions": [
            {"label": "hyaena:left", "species": "hyaena", "viewpoint": "left", "probability": 0.99}
        ]
    }

    def get_model(model_id):
        if model_id == "detect-mock":
            return detect_model
        if model_id == "labeler-mock":
            return labeler_model
        return None

    handler.get_model = get_model
    app.state.model_handler = handler

    # Clear job store between tests
    with _jobs_lock:
        _jobs.clear()

    with TestClient(app) as c:
        yield c


HYENA_IMAGE = "/mnt/c/claude-skills/datasets/hyena.coco/images/train2022/000000002135.jpg"


def _wait_for_job(client, jobid, timeout=120):
    """Poll job status until completed or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        resp = client.get("/api/engine/job/status/", params={"jobid": jobid})
        status = resp.json()["response"]["jobstatus"]
        if status in ("completed", "exception"):
            return status
        time.sleep(0.2)
    raise TimeoutError(f"Job {jobid} did not complete in {timeout}s")


# ---------------------------------------------------------------------------
# Unit tests: Response envelope
# ---------------------------------------------------------------------------

class TestWbiaResponseEnvelope:

    def test_success_envelope(self):
        r = _wbia_response({"foo": "bar"})
        assert r["status"]["success"] is True
        assert r["status"]["code"] == ""
        assert r["status"]["message"] == ""
        assert r["status"]["cache"] == -1
        assert r["response"] == {"foo": "bar"}

    def test_error_envelope(self):
        r = _wbia_response(None, success=False, message="Something failed")
        assert r["status"]["success"] is False
        assert r["status"]["message"] == "Something failed"
        assert r["response"] is None

    def test_string_response(self):
        """Jobid is returned as a string in the response field."""
        r = _wbia_response("some-uuid-string")
        assert r["response"] == "some-uuid-string"


class TestStripInternal:

    def test_strips_underscore_prefix(self):
        det = {"id": 1, "xtl": 10, "_bbox": [10, 20, 30, 40], "_internal": True}
        result = _strip_internal(det)
        assert "_bbox" not in result
        assert "_internal" not in result
        assert result["id"] == 1
        assert result["xtl"] == 10

    def test_no_internal_fields(self):
        det = {"id": 1, "species": "lion"}
        result = _strip_internal(det)
        assert result == det


# ---------------------------------------------------------------------------
# Unit tests: Job queue
# ---------------------------------------------------------------------------

class TestJobQueue:

    def test_set_and_get_job(self):
        with _jobs_lock:
            _jobs.clear()
        _set_job("test-1", "received")
        job = _get_job("test-1")
        assert job["status"] == "received"

    def test_result_set_before_status(self):
        """Result must be available when status is 'completed'."""
        with _jobs_lock:
            _jobs.clear()
        _set_job("test-2", "completed", result={"data": "here"})
        job = _get_job("test-2")
        assert job["status"] == "completed"
        assert job["result"] == {"data": "here"}

    def test_get_nonexistent_job(self):
        job = _get_job("nonexistent-job-id")
        assert job is None

    def test_job_status_endpoint_unknown(self, mock_client):
        resp = mock_client.get("/api/engine/job/status/", params={"jobid": "nope"})
        assert resp.status_code == 200
        assert resp.json()["response"]["jobstatus"] == "unknown"

    def test_job_result_not_found(self, mock_client):
        resp = mock_client.get("/api/engine/job/result/", params={"jobid": "nope"})
        assert resp.status_code == 200
        assert resp.json()["status"]["success"] is False

    def test_job_list(self, mock_client):
        resp = mock_client.get("/api/engine/job/")
        assert resp.status_code == 200
        assert isinstance(resp.json()["response"], list)

    def test_thread_safety(self):
        """Concurrent job updates should not crash."""
        with _jobs_lock:
            _jobs.clear()
        errors = []

        def writer(n):
            try:
                for i in range(50):
                    _set_job(f"thread-{n}-job-{i}", "working")
                    _set_job(f"thread-{n}-job-{i}", "completed", result={"i": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# Unit tests: Detection with mocks
# ---------------------------------------------------------------------------

class TestDetectionMocked:

    def test_basic_detection(self, mock_client):
        resp = mock_client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": [HYENA_IMAGE],
            "model_tag": "detect-mock",
        })
        assert resp.status_code == 200
        jobid = resp.json()["response"]

        status = _wait_for_job(mock_client, jobid, timeout=10)
        assert status == "completed"

        resp = mock_client.get("/api/engine/job/result/", params={"jobid": jobid})
        result = resp.json()["response"]["json_result"]

        assert len(result["results_list"]) == 1
        annots = result["results_list"][0]
        assert len(annots) == 2

        # Verify annotation format
        a = annots[0]
        assert a["xtl"] == 10
        assert a["ytl"] == 20
        assert a["width"] == 100
        assert a["height"] == 80
        assert a["confidence"] == 0.95
        assert a["class"] == "hyaena"
        assert a["species"] == "hyaena"
        assert "_bbox" not in a  # Internal field stripped

    def test_sensitivity_filter(self, mock_client):
        resp = mock_client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": [HYENA_IMAGE],
            "model_tag": "detect-mock",
            "sensitivity": 0.8,
        })
        jobid = resp.json()["response"]
        _wait_for_job(mock_client, jobid, timeout=10)

        resp = mock_client.get("/api/engine/job/result/", params={"jobid": jobid})
        result = resp.json()["response"]["json_result"]
        annots = result["results_list"][0]
        # Only the 0.95 detection should remain (0.70 filtered out)
        assert len(annots) == 1
        assert annots[0]["confidence"] == 0.95

    def test_detection_with_labeling(self, mock_client):
        resp = mock_client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": [HYENA_IMAGE],
            "model_tag": "detect-mock",
            "labeler_model_tag": "labeler-mock",
            "use_labeler_species": True,
        })
        jobid = resp.json()["response"]
        _wait_for_job(mock_client, jobid, timeout=10)

        resp = mock_client.get("/api/engine/job/result/", params={"jobid": jobid})
        result = resp.json()["response"]["json_result"]
        annots = result["results_list"][0]

        for a in annots:
            assert a["viewpoint"] == "left"
            assert a["species"] == "hyaena"

    def test_missing_model(self, mock_client):
        resp = mock_client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": [HYENA_IMAGE],
            "model_tag": "nonexistent",
        })
        jobid = resp.json()["response"]
        _wait_for_job(mock_client, jobid, timeout=10)

        resp = mock_client.get("/api/engine/job/result/", params={"jobid": jobid})
        result = resp.json()["response"]["json_result"]
        assert result["results_list"] == [[]]

    def test_no_model_tag(self, mock_client):
        resp = mock_client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": [HYENA_IMAGE],
        })
        jobid = resp.json()["response"]
        _wait_for_job(mock_client, jobid, timeout=10)

        resp = mock_client.get("/api/engine/job/result/", params={"jobid": jobid})
        result = resp.json()["response"]["json_result"]
        assert result["results_list"] == [[]]

    def test_bad_image_path(self, mock_client):
        resp = mock_client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": ["/nonexistent/image.jpg"],
            "model_tag": "detect-mock",
        })
        jobid = resp.json()["response"]
        _wait_for_job(mock_client, jobid, timeout=10)

        resp = mock_client.get("/api/engine/job/result/", params={"jobid": jobid})
        result = resp.json()["response"]["json_result"]
        assert result["results_list"] == [[]]

    def test_multiple_images(self, mock_client):
        resp = mock_client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": [HYENA_IMAGE, HYENA_IMAGE],
            "model_tag": "detect-mock",
        })
        jobid = resp.json()["response"]
        _wait_for_job(mock_client, jobid, timeout=10)

        resp = mock_client.get("/api/engine/job/result/", params={"jobid": jobid})
        result = resp.json()["response"]["json_result"]
        assert len(result["results_list"]) == 2
        assert len(result["image_uuid_list"]) == 2
        assert len(result["score_list"]) == 2

    def test_all_wbia_annotation_fields(self, mock_client):
        """Every annotation must have all fields WBIA returns."""
        resp = mock_client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": [HYENA_IMAGE],
            "model_tag": "detect-mock",
        })
        jobid = resp.json()["response"]
        _wait_for_job(mock_client, jobid, timeout=10)

        resp = mock_client.get("/api/engine/job/result/", params={"jobid": jobid})
        annot = resp.json()["response"]["json_result"]["results_list"][0][0]

        required_fields = [
            "id", "uuid", "xtl", "ytl", "left", "top", "width", "height",
            "theta", "confidence", "class", "species", "viewpoint",
            "quality", "multiple", "interest",
        ]
        for field in required_fields:
            assert field in annot, f"Missing WBIA field: {field}"

    def test_viewpoint_model_tag_fallback(self, mock_client):
        """viewpoint_model_tag should work as fallback for labeler_model_tag."""
        resp = mock_client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": [HYENA_IMAGE],
            "model_tag": "detect-mock",
            "viewpoint_model_tag": "labeler-mock",
        })
        jobid = resp.json()["response"]
        _wait_for_job(mock_client, jobid, timeout=10)

        resp = mock_client.get("/api/engine/job/result/", params={"jobid": jobid})
        annot = resp.json()["response"]["json_result"]["results_list"][0][0]
        assert annot["viewpoint"] == "left"

    def test_yolo_endpoint_alias(self, mock_client):
        resp = mock_client.post("/api/engine/detect/cnn/yolo/", json={
            "image_uuid_list": [HYENA_IMAGE],
            "model_tag": "detect-mock",
        })
        assert resp.status_code == 200
        jobid = resp.json()["response"]
        status = _wait_for_job(mock_client, jobid, timeout=10)
        assert status == "completed"

    def test_lightnet_endpoint_alias(self, mock_client):
        resp = mock_client.post("/api/engine/detect/cnn/lightnet/", json={
            "image_uuid_list": [HYENA_IMAGE],
            "model_tag": "detect-mock",
        })
        assert resp.status_code == 200
        jobid = resp.json()["response"]
        status = _wait_for_job(mock_client, jobid, timeout=10)
        assert status == "completed"

    def test_has_assignments_false_without_assigner(self, mock_client):
        resp = mock_client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": [HYENA_IMAGE],
            "model_tag": "detect-mock",
        })
        jobid = resp.json()["response"]
        _wait_for_job(mock_client, jobid, timeout=10)

        resp = mock_client.get("/api/engine/job/result/", params={"jobid": jobid})
        result = resp.json()["response"]["json_result"]
        assert result["has_assignments"] is False

    def test_empty_thetas_from_model(self, mock_client):
        """Models returning empty thetas list should not drop detections."""
        app = mock_client.app
        handler = app.state.model_handler
        model = handler.get_model("detect-mock")
        original_return = model.predict.return_value.copy()

        # Simulate model returning empty thetas
        model.predict.return_value = {
            "bboxes": [[10, 20, 100, 80]],
            "scores": [0.95],
            "class_names": ["hyaena"],
            "thetas": [],
        }

        resp = mock_client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": [HYENA_IMAGE],
            "model_tag": "detect-mock",
        })
        jobid = resp.json()["response"]
        _wait_for_job(mock_client, jobid, timeout=10)

        resp = mock_client.get("/api/engine/job/result/", params={"jobid": jobid})
        result = resp.json()["response"]["json_result"]
        annots = result["results_list"][0]
        assert len(annots) == 1
        assert annots[0]["theta"] == 0.0

        # Restore
        model.predict.return_value = original_return


# ---------------------------------------------------------------------------
# Unit tests: Labeler stub endpoint
# ---------------------------------------------------------------------------

class TestLabelerStub:

    def test_returns_not_implemented(self, mock_client):
        resp = mock_client.post("/api/engine/labeler/cnn/", json={
            "annot_uuid_list": ["some-uuid"],
        })
        assert resp.status_code == 200
        jobid = resp.json()["response"]

        resp = mock_client.get("/api/engine/job/result/", params={"jobid": jobid})
        result = resp.json()["response"]["json_result"]
        assert result["status"] == "not_implemented"


# ---------------------------------------------------------------------------
# Integration tests: Real models
# ---------------------------------------------------------------------------

class TestRealModels:

    def test_detection_only(self, real_client):
        resp = real_client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": [HYENA_IMAGE],
            "model_tag": "detect-hyaena",
            "sensitivity": 0.3,
        })
        jobid = resp.json()["response"]
        status = _wait_for_job(real_client, jobid)
        assert status == "completed"

        resp = real_client.get("/api/engine/job/result/", params={"jobid": jobid})
        result = resp.json()["response"]["json_result"]
        annots = result["results_list"][0]
        assert len(annots) >= 1
        assert annots[0]["class"] == "hyaena"
        assert annots[0]["confidence"] > 0.5

    def test_detection_with_labeler(self, real_client):
        resp = real_client.post("/api/engine/detect/cnn/lightnet/", json={
            "image_uuid_list": [HYENA_IMAGE],
            "model_tag": "detect-hyaena",
            "labeler_model_tag": "labeler-hyaena",
            "sensitivity": 0.3,
            "use_labeler_species": True,
        })
        jobid = resp.json()["response"]
        status = _wait_for_job(real_client, jobid)
        assert status == "completed"

        resp = real_client.get("/api/engine/job/result/", params={"jobid": jobid})
        result = resp.json()["response"]["json_result"]
        annots = result["results_list"][0]
        assert len(annots) >= 1
        assert annots[0]["viewpoint"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

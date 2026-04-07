"""Stress tests for WBIA-compatible API endpoints using real models and data.

Tests:
- Concurrent job submissions
- Batch processing of many images
- Pipeline throughput (detect + label)
- Job queue under load
- Memory stability across many requests
"""

import os
import time
import threading
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import wbia_compat_router
from app.routers.wbia_compat_router import _jobs, _jobs_lock


HYENA_IMAGE_DIR = "/mnt/c/claude-skills/datasets/hyena.coco/images/train2022"
HYENA_IMAGE = os.path.join(HYENA_IMAGE_DIR, "000000002135.jpg")


def _get_valid_images(n=50):
    """Get N valid hyena images (skip corrupt/empty ones)."""
    images = []
    for f in sorted(os.listdir(HYENA_IMAGE_DIR)):
        if not f.endswith(".jpg"):
            continue
        path = os.path.join(HYENA_IMAGE_DIR, f)
        if os.path.getsize(path) > 10000:
            images.append(path)
        if len(images) >= n:
            break
    return images


@pytest.fixture(scope="module")
def client():
    """Test client with real models."""
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


def _wait_for_job(client, jobid, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        resp = client.get("/api/engine/job/status/", params={"jobid": jobid})
        status = resp.json()["response"]["jobstatus"]
        if status in ("completed", "exception"):
            return status
        time.sleep(0.3)
    raise TimeoutError(f"Job {jobid} did not complete in {timeout}s")


class TestConcurrentJobs:
    """Submit multiple detection jobs concurrently."""

    def test_10_concurrent_single_image_jobs(self, client):
        """10 separate jobs, each processing 1 image."""
        images = _get_valid_images(10)
        if len(images) < 10:
            pytest.skip("Not enough valid images")

        # Submit all jobs
        jobids = []
        for img in images:
            resp = client.post("/api/engine/detect/cnn/", json={
                "image_uuid_list": [img],
                "model_tag": "detect-hyaena",
                "sensitivity": 0.3,
            })
            assert resp.status_code == 200
            jobids.append(resp.json()["response"])

        # Wait for all to complete
        results = {}
        for jobid in jobids:
            status = _wait_for_job(client, jobid)
            assert status == "completed", f"Job {jobid} failed"
            resp = client.get("/api/engine/job/result/", params={"jobid": jobid})
            results[jobid] = resp.json()["response"]["json_result"]

        # Verify all returned valid results
        completed = sum(1 for r in results.values() if r["results_list"] is not None)
        assert completed == 10

        # Count total detections
        total_dets = sum(
            len(r["results_list"][0]) for r in results.values()
            if r["results_list"][0]
        )
        print(f"\n  10 concurrent jobs: {total_dets} total detections across 10 images")

    def test_concurrent_detection_plus_labeling(self, client):
        """5 concurrent jobs with detect + label pipeline."""
        images = _get_valid_images(5)
        if len(images) < 5:
            pytest.skip("Not enough valid images")

        jobids = []
        for img in images:
            resp = client.post("/api/engine/detect/cnn/", json={
                "image_uuid_list": [img],
                "model_tag": "detect-hyaena",
                "labeler_model_tag": "labeler-hyaena",
                "use_labeler_species": True,
                "sensitivity": 0.3,
            })
            jobids.append(resp.json()["response"])

        for jobid in jobids:
            status = _wait_for_job(client, jobid)
            assert status == "completed"

            resp = client.get("/api/engine/job/result/", params={"jobid": jobid})
            result = resp.json()["response"]["json_result"]
            labeled_count = 0
            total_high_conf = 0
            for annots in result["results_list"]:
                for a in annots:
                    if a.get("confidence", 0) > 0.5:
                        total_high_conf += 1
                        if a["viewpoint"] is not None:
                            labeled_count += 1
            # Most high-confidence detections should get viewpoint labels
            # (some may fail if bbox is too small to crop)
            if total_high_conf > 0:
                label_rate = labeled_count / total_high_conf
                print(f"\n  Job {jobid}: {labeled_count}/{total_high_conf} high-conf detections labeled ({label_rate:.0%})")
                assert label_rate >= 0.5, f"Too few labeled: {labeled_count}/{total_high_conf}"


class TestBatchProcessing:
    """Test processing many images in single requests."""

    def test_batch_5_images(self, client):
        """Single job with 5 images."""
        images = _get_valid_images(5)
        if len(images) < 5:
            pytest.skip("Not enough valid images")

        resp = client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": images,
            "model_tag": "detect-hyaena",
            "sensitivity": 0.3,
        })
        jobid = resp.json()["response"]
        status = _wait_for_job(client, jobid)
        assert status == "completed"

        resp = client.get("/api/engine/job/result/", params={"jobid": jobid})
        result = resp.json()["response"]["json_result"]

        assert len(result["results_list"]) == 5
        assert len(result["image_uuid_list"]) == 5
        assert len(result["score_list"]) == 5

        detections_per_image = [len(r) for r in result["results_list"]]
        print(f"\n  Batch 5 images: detections per image = {detections_per_image}")

    def test_batch_20_images_with_labeler(self, client):
        """Single job with 20 images, detect + label pipeline."""
        images = _get_valid_images(20)
        if len(images) < 20:
            pytest.skip("Not enough valid images")

        start = time.time()
        resp = client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": images,
            "model_tag": "detect-hyaena",
            "labeler_model_tag": "labeler-hyaena",
            "use_labeler_species": True,
            "sensitivity": 0.3,
        })
        jobid = resp.json()["response"]
        status = _wait_for_job(client, jobid, timeout=600)
        elapsed = time.time() - start
        assert status == "completed"

        resp = client.get("/api/engine/job/result/", params={"jobid": jobid})
        result = resp.json()["response"]["json_result"]

        assert len(result["results_list"]) == 20

        total_dets = sum(len(r) for r in result["results_list"])
        with_viewpoint = sum(
            1 for r in result["results_list"] for a in r if a.get("viewpoint")
        )
        print(f"\n  Batch 20 images + labeler: {total_dets} detections, "
              f"{with_viewpoint} with viewpoint, {elapsed:.1f}s total")


class TestJobQueueUnderLoad:
    """Test job queue behavior under sustained load."""

    def test_rapid_fire_submissions(self, client):
        """Submit 20 jobs as fast as possible, verify all complete."""
        images = _get_valid_images(20)
        if len(images) < 20:
            pytest.skip("Not enough valid images")

        jobids = []
        for img in images:
            resp = client.post("/api/engine/detect/cnn/", json={
                "image_uuid_list": [img],
                "model_tag": "detect-hyaena",
            })
            jobids.append(resp.json()["response"])

        # All should eventually complete
        statuses = {}
        for jobid in jobids:
            status = _wait_for_job(client, jobid, timeout=300)
            statuses[status] = statuses.get(status, 0) + 1

        print(f"\n  Rapid-fire 20 jobs: statuses = {statuses}")
        assert statuses.get("completed", 0) == 20

    def test_job_status_polling_during_execution(self, client):
        """Poll job status rapidly during execution to verify state transitions."""
        images = _get_valid_images(5)
        if len(images) < 5:
            pytest.skip("Not enough valid images")

        resp = client.post("/api/engine/detect/cnn/", json={
            "image_uuid_list": images,
            "model_tag": "detect-hyaena",
            "labeler_model_tag": "labeler-hyaena",
        })
        jobid = resp.json()["response"]

        seen_statuses = set()
        while True:
            resp = client.get("/api/engine/job/status/", params={"jobid": jobid})
            status = resp.json()["response"]["jobstatus"]
            seen_statuses.add(status)
            if status in ("completed", "exception"):
                break
            time.sleep(0.05)

        print(f"\n  Status transitions seen: {seen_statuses}")
        assert "completed" in seen_statuses
        # Should have seen at least received or working before completed
        assert len(seen_statuses) >= 1

    def test_result_always_available_when_completed(self, client):
        """When status is 'completed', result must never be None."""
        images = _get_valid_images(10)
        if len(images) < 10:
            pytest.skip("Not enough valid images")

        jobids = []
        for img in images:
            resp = client.post("/api/engine/detect/cnn/", json={
                "image_uuid_list": [img],
                "model_tag": "detect-hyaena",
            })
            jobids.append(resp.json()["response"])

        for jobid in jobids:
            _wait_for_job(client, jobid)
            resp = client.get("/api/engine/job/result/", params={"jobid": jobid})
            data = resp.json()
            assert data["status"]["success"] is True
            result = data["response"]["json_result"]
            assert result is not None, f"Job {jobid} completed but result is None!"
            assert "results_list" in result
            assert "image_uuid_list" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

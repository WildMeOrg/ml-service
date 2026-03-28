#!/usr/bin/env python3
"""
Stress test suite for ML inference service.
Tests sustained load, burst load, mixed load, and memory stability.
Target: 50,000+ detections and extractions per day (~0.6 req/sec sustained).
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class StressTestResult:
    """Stores results from a stress test run."""
    scenario: str
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    latencies: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def error_rate(self):
        return (self.failed / self.total_requests * 100) if self.total_requests else 0

    @property
    def throughput(self):
        duration = self.end_time - self.start_time
        return self.total_requests / duration if duration > 0 else 0

    def percentile(self, p):
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def summary(self):
        duration = self.end_time - self.start_time
        print(f"\n{'=' * 60}")
        print(f"Stress Test Results: {self.scenario}")
        print(f"{'=' * 60}")
        print(f"Total requests:    {self.total_requests}")
        print(f"Successful:        {self.successful}")
        print(f"Failed:            {self.failed}")
        print(f"Error rate:        {self.error_rate:.1f}%")
        print(f"Duration:          {duration:.1f}s")
        print(f"Throughput:        {self.throughput:.2f} req/sec")
        if self.latencies:
            print(f"Latency p50:       {self.percentile(50):.3f}s")
            print(f"Latency p95:       {self.percentile(95):.3f}s")
            print(f"Latency p99:       {self.percentile(99):.3f}s")
            print(f"Latency mean:      {statistics.mean(self.latencies):.3f}s")
            print(f"Latency max:       {max(self.latencies):.3f}s")
        if self.errors:
            print(f"\nFirst 5 errors:")
            for err in self.errors[:5]:
                print(f"  - {err}")
        print(f"{'=' * 60}")


async def send_request(client, url, payload, result: StressTestResult):
    """Send a single request and record the result."""
    start = time.monotonic()
    try:
        response = await client.post(url, json=payload, timeout=120.0)
        elapsed = time.monotonic() - start
        result.total_requests += 1
        result.latencies.append(elapsed)
        if response.status_code == 200:
            result.successful += 1
        else:
            result.failed += 1
            result.errors.append(f"HTTP {response.status_code}: {response.text[:200]}")
    except Exception as e:
        elapsed = time.monotonic() - start
        result.total_requests += 1
        result.failed += 1
        result.latencies.append(elapsed)
        result.errors.append(str(e)[:200])


async def run_sustained(base_url: str, image_path: str, num_requests: int = 100):
    """Sustained load: sequential requests at ~1 req/sec."""
    result = StressTestResult(scenario="Sustained Load")

    predict_url = f"{base_url}/predict/"
    predict_payload = {
        "model_id": "msv3",
        "image_uri": image_path,
    }

    extract_url = f"{base_url}/extract/"
    extract_payload = {
        "model_id": "miewid-msv4.1",
        "image_uri": image_path,
        "bbox": [50, 50, 200, 200],
        "theta": 0.0,
    }

    print(f"Running sustained load test: {num_requests} requests at ~1 req/sec")
    result.start_time = time.monotonic()

    async with httpx.AsyncClient() as client:
        for i in range(num_requests):
            # Alternate between predict and extract
            if i % 2 == 0:
                await send_request(client, predict_url, predict_payload, result)
            else:
                await send_request(client, extract_url, extract_payload, result)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{num_requests} "
                      f"(ok={result.successful}, fail={result.failed})")

            # Pace to ~1 req/sec
            await asyncio.sleep(1.0)

    result.end_time = time.monotonic()
    result.summary()
    return result


async def run_burst(base_url: str, image_path: str, concurrency: int = 10, total: int = 50):
    """Burst load: many concurrent requests."""
    result = StressTestResult(scenario=f"Burst Load (concurrency={concurrency})")

    predict_url = f"{base_url}/predict/"
    predict_payload = {
        "model_id": "msv3",
        "image_uri": image_path,
    }

    extract_url = f"{base_url}/extract/"
    extract_payload = {
        "model_id": "miewid-msv4.1",
        "image_uri": image_path,
        "bbox": [50, 50, 200, 200],
        "theta": 0.0,
    }

    print(f"Running burst load test: {total} requests, {concurrency} concurrent")
    result.start_time = time.monotonic()

    sem = asyncio.Semaphore(concurrency)

    async def bounded_request(client, url, payload):
        async with sem:
            await send_request(client, url, payload, result)

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(total):
            if i % 2 == 0:
                tasks.append(bounded_request(client, predict_url, predict_payload))
            else:
                tasks.append(bounded_request(client, extract_url, extract_payload))
        await asyncio.gather(*tasks)

    result.end_time = time.monotonic()
    result.summary()
    return result


async def run_mixed(base_url: str, image_path: str, num_requests: int = 50):
    """Mixed load: interleaved predict, extract, and classify requests."""
    result = StressTestResult(scenario="Mixed Load")

    endpoints = [
        (f"{base_url}/predict/", {
            "model_id": "msv3",
            "image_uri": image_path,
        }),
        (f"{base_url}/extract/", {
            "model_id": "miewid-msv4.1",
            "image_uri": image_path,
            "bbox": [50, 50, 200, 200],
            "theta": 0.0,
        }),
    ]

    print(f"Running mixed load test: {num_requests} interleaved requests")
    result.start_time = time.monotonic()

    sem = asyncio.Semaphore(5)

    async def bounded_request(client, url, payload):
        async with sem:
            await send_request(client, url, payload, result)

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(num_requests):
            url, payload = endpoints[i % len(endpoints)]
            tasks.append(bounded_request(client, url, payload))
        await asyncio.gather(*tasks)

    result.end_time = time.monotonic()
    result.summary()
    return result


async def run_memory(base_url: str, image_path: str, num_requests: int = 500):
    """Memory stability: many sequential requests to check for GPU memory growth."""
    result = StressTestResult(scenario="Memory Stability")

    extract_url = f"{base_url}/extract/"
    extract_payload = {
        "model_id": "miewid-msv4.1",
        "image_uri": image_path,
        "bbox": [50, 50, 200, 200],
        "theta": 0.0,
    }

    print(f"Running memory stability test: {num_requests} sequential extract requests")
    print("Monitor GPU memory externally with: watch -n 1 nvidia-smi")
    result.start_time = time.monotonic()

    async with httpx.AsyncClient() as client:
        for i in range(num_requests):
            await send_request(client, extract_url, extract_payload, result)

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{num_requests} "
                      f"(ok={result.successful}, fail={result.failed}, "
                      f"p50={result.percentile(50):.3f}s)")

    result.end_time = time.monotonic()
    result.summary()
    return result


def main():
    parser = argparse.ArgumentParser(description="Stress test the ML inference service")
    parser.add_argument("--base-url", default="http://localhost:6050",
                        help="Base URL of the service")
    parser.add_argument("--image", default="Images/img1.png",
                        help="Path to test image (as seen by the service)")
    parser.add_argument("--scenario", required=True,
                        choices=["sustained", "burst", "mixed", "memory", "all"],
                        help="Test scenario to run")
    parser.add_argument("--requests", type=int, default=None,
                        help="Number of requests (overrides scenario default)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Concurrency level for burst test")
    args = parser.parse_args()

    scenarios = {
        "sustained": lambda: run_sustained(args.base_url, args.image,
                                           args.requests or 100),
        "burst": lambda: run_burst(args.base_url, args.image,
                                   args.concurrency, args.requests or 50),
        "mixed": lambda: run_mixed(args.base_url, args.image,
                                   args.requests or 50),
        "memory": lambda: run_memory(args.base_url, args.image,
                                     args.requests or 500),
    }

    if args.scenario == "all":
        results = []
        for name, run_fn in scenarios.items():
            print(f"\n{'#' * 60}")
            print(f"# Running scenario: {name}")
            print(f"{'#' * 60}")
            result = asyncio.run(run_fn())
            results.append(result)

        print(f"\n\n{'=' * 60}")
        print("OVERALL SUMMARY")
        print(f"{'=' * 60}")
        all_passed = True
        for r in results:
            status = "PASS" if r.error_rate == 0 else "FAIL"
            if r.error_rate > 0:
                all_passed = False
            print(f"  {r.scenario}: {status} "
                  f"({r.successful}/{r.total_requests}, "
                  f"p95={r.percentile(95):.3f}s)")
        sys.exit(0 if all_passed else 1)
    else:
        result = asyncio.run(scenarios[args.scenario]())
        sys.exit(0 if result.error_rate == 0 else 1)


if __name__ == "__main__":
    main()

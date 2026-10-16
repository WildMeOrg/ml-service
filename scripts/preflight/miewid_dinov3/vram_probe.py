"""Measure the GPU memory the DINOv3 MiewID adds to a production registry.

Run INSIDE the production image on the production GPU, from the repo root:
  python3 -m scripts.preflight.miewid_dinov3.vram_probe \
      --config /path/to/prod/model_config.json --dinov3-entry dinov3_entry.json

Prints the baseline with every prod model loaded, the increment from loading
the DINOv3 entry, and the peak increment during one DINOv3 extraction.
"""
import argparse
import io
import json

import numpy as np
import torch
from PIL import Image

from app.models.model_handler import ModelHandler

MiB = 2 ** 20


def load(handler, entry):
    params = {k: v for k, v in entry.items() if k not in ("model_id", "model_type")}
    handler.load_model(model_id=entry["model_id"], model_type=entry["model_type"],
                       device="cuda", **params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the production model_config.json")
    ap.add_argument("--dinov3-entry", required=True, help="JSON file with one DINOv3 model entry")
    args = ap.parse_args()

    handler = ModelHandler()
    for entry in json.load(open(args.config))["models"]:
        load(handler, entry)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()

    dino_entry = json.load(open(args.dinov3_entry))
    load(handler, dino_entry)
    torch.cuda.synchronize()
    loaded = torch.cuda.memory_allocated()

    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (3000, 4000, 3), dtype=np.uint8)
                    ).save(buf, format="JPEG")
    torch.cuda.reset_peak_memory_stats()
    handler.get_model(dino_entry["model_id"]).extract_embeddings(buf.getvalue(), bbox=(100, 100, 2000, 1500))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    total = torch.cuda.get_device_properties(0).total_memory
    device_free, _ = torch.cuda.mem_get_info()
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(0),
        "total_mib": round(total / MiB),
        "baseline_all_prod_models_mib": round(baseline / MiB),
        "dinov3_resident_increment_mib": round((loaded - baseline) / MiB),
        "dinov3_extract_peak_increment_mib": round((peak - baseline) / MiB),
        "torch_reserved_mib": round(torch.cuda.memory_reserved() / MiB),
        # What the driver says is free right now: includes the CUDA context,
        # library workspaces and every other process on the GPU.
        "device_free_mib": round(device_free / MiB),
        # total - PyTorch peak; EXCLUDES the context, libraries and other
        # processes, so it overstates what is actually available.
        "theoretical_headroom_after_peak_mib": round((total - peak) / MiB),
    }, indent=2))


if __name__ == "__main__":
    main()

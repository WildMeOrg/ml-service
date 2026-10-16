"""Service half of the MiewID DINOv3 parity gate. Runs INSIDE the production
image on the production GPU, from the repository root:

  python3 -m scripts.preflight.miewid_dinov3.run_gate ...

  arch: the reference's own preprocessed tensors -> DinoV3MiewIdNet.forward_stages
        (tokens, concat, embedding), under the service's autocast
  e2e:  the same image FILE BYTES (sha256-checked) ->
        MiewidModel.extract_embeddings, per bbox mode. The chip and the
        input tensor are CAPTURED from the service path (preprocess wrapper,
        forward pre-hook), never recomputed here.
  retrieval: ordered top-5 over the chip set; cosine distance per chip
  negative control: e2e embeddings x1.05 must FAIL the raw check

Exit 0 only if every check passes in both bbox modes AND every negative
control fails, and thresholds_status is "agreed" (or --allow-proposed).
Any input misalignment (row set/order vs manifest expected_chips, row
metadata, image bytes, checkpoint) aborts before inference.

Rows listed in manifest known_divergence (overhanging bboxes, where the service
and training chippers differ) are computed and REPORTED in every e2e stage but
excluded from the gated e2e stages, e2e top-5 and the e2e negative control.
The arch stage stays gated for all rows.
"""
import argparse
import ast
import csv
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

from app.models.miewid import MiewidModel
from scripts.preflight.miewid_dinov3 import metrics

ROW_KEYS = ("case", "species", "name", "bbox", "theta")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rewrite(path, rules):
    for old, new in rules:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def check_alignment(rows, ref_meta, ref, paths):
    n = len(rows)
    if n != len(ref_meta["chips"]):
        sys.exit(f"{n} CSV rows vs {len(ref_meta['chips'])} reference chips")
    for i, (row, meta) in enumerate(zip(rows, ref_meta["chips"])):
        for key in ROW_KEYS:
            if str(row[key]) != str(meta[key]):
                sys.exit(f"row {i}: {key} {row[key]!r} != reference {meta[key]!r}")
        got = sha256_file(paths[i])
        if got != meta["image_sha256"]:
            sys.exit(f"row {i}: image bytes differ from the reference ({paths[i]})")
    expected = {"input": (n, 3, 448, 448), "tokens": (n, 789, 768),
                "concat": (n, 1536), "embedding": (n, 1536)}
    for mode in ("router", "float"):
        for k, shape in expected.items():
            if ref[f"{mode}_{k}"].shape != shape:
                sys.exit(f"reference {mode}_{k} has shape {ref[f'{mode}_{k}'].shape}, expected {shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--reference", required=True, help="dir written by reference_runner.py")
    ap.add_argument("--path-column", default="host_path")
    ap.add_argument("--rewrite-prefix", action="append", default=[], help="OLD=NEW")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--allow-proposed", action="store_true",
                    help="exit 0 on a pass even though thresholds_status is not 'agreed'")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    th = manifest["thresholds"]
    if manifest["checkpoint"]["sha256"].startswith("<"):
        sys.exit("manifest checkpoint sha256 is still a placeholder; fill it from the training session")
    ref_dir = pathlib.Path(args.reference)
    ref = np.load(ref_dir / "reference.npz")
    ref_meta = json.load(open(ref_dir / "reference.json"))
    rows = list(csv.DictReader(open(ref_dir / "reference_chips.csv")))
    problems = (metrics.check_expected_chips(rows, manifest["expected_chips"])
                + metrics.check_known_divergence(manifest["known_divergence"], manifest["expected_chips"]))
    if problems:
        sys.exit("reference gallery does not match the manifest:\n  " + "\n  ".join(problems))
    sel = metrics.select_rows(manifest["expected_chips"], manifest["known_divergence"])
    gated, reported = sel["gated"], sel["reported"]
    rules = [tuple(r.split("=", 1)) for r in args.rewrite_prefix]
    paths = [rewrite(row[args.path_column], rules) for row in rows]
    check_alignment(rows, ref_meta, ref, paths)

    m = MiewidModel()
    m.load(device="cuda", model_id="gate", backbone="dinov3_vitb16",
           checkpoint_path=manifest["checkpoint"]["path"],
           checkpoint_sha256=manifest["checkpoint"]["sha256"], version=5)
    if m.checkpoint_sha256 != ref_meta["checkpoint_sha256"]:
        sys.exit(f"service checkpoint {m.checkpoint_sha256} != reference {ref_meta['checkpoint_sha256']}")

    # Capture what the service actually feeds: the chip handed to preprocess
    # and the tensor handed to the network.
    captured = {}
    original_preprocess = m.preprocess

    def capturing_preprocess(image):
        captured["chip"] = np.array(image, copy=True)
        return original_preprocess(image=image)

    def capture_input(mod, args_):
        captured["input"] = args_[0][0].float().cpu().numpy()

    m.preprocess = capturing_preprocess
    m.model.register_forward_pre_hook(capture_input)

    results, controls, per_chip, known_report = [], [], [], []
    for mode in ("router", "float"):
        # arch: identical tensors in
        arch = {"tokens": [], "concat": [], "embedding": []}
        for x in ref[f"{mode}_input"]:
            with torch.no_grad(), m._autocast():
                stages = m.model.forward_stages(torch.from_numpy(x).unsqueeze(0).cuda())
            for k in arch:
                arch[k].append(stages[k][0].float().cpu().numpy())
        for k, v in arch.items():
            results.append(metrics.check_stage(f"{mode}.arch.{k}", np.stack(v), ref[f"{mode}_{k}"], th["arch"][k]))

        # e2e: identical bytes in, for every row; only `gated` rows are judged
        chips, inputs, embs = [], [], []
        for i, row in enumerate(rows):
            raw = pathlib.Path(paths[i]).read_bytes()
            bbox = ast.literal_eval(row["bbox"])
            bbox = metrics.router_bbox(bbox) if mode == "router" else tuple(bbox)
            captured.clear()
            embs.append(m.extract_embeddings(raw, bbox=bbox, theta=float(row["theta"]))[0])
            chips.append(metrics.check_chip(captured["chip"], ref[f"{mode}_chip_{i}"], th["e2e"]["chip"]))
            inputs.append(captured["input"])
        embs, inputs = np.stack(embs), np.stack(inputs)
        ref_emb, ref_in = ref[f"{mode}_embedding"], ref[f"{mode}_input"]
        cos = [metrics.cosine_distance(a, b) for a, b in zip(embs, ref_emb)]

        chips_ok = [{"index": i, "case": rows[i]["case"], **chips[i]} for i in gated]
        results.append({"stage": f"{mode}.e2e.chip", "pass": all(c["pass"] for c in chips_ok),
                        "per_chip": chips_ok})
        results.append(metrics.check_stage(f"{mode}.e2e.input", inputs[gated], ref_in[gated], th["e2e"]["input"]))
        results.append(metrics.check_stage(f"{mode}.e2e.embedding", embs[gated], ref_emb[gated], th["e2e"]["embedding"]))
        results.append({"stage": f"{mode}.e2e.cosine",
                        **metrics.cosine_gate([cos[i] for i in gated], th["cosine_distance_max"])})
        top5 = metrics.topk_agreement(embs[gated], ref_emb[gated], 5)
        results.append({"stage": f"{mode}.e2e.top5", "agreement": top5, "pass": top5 >= th["top5_agreement_min"]})
        per_chip += [{"mode": mode, "index": i, "case": rows[i]["case"], "species": rows[i]["species"],
                      "cosine_distance": cos[i], "gated": i in gated} for i in range(len(rows))]

        # Known divergence: every e2e stage reported, none gated.
        for i in reported:
            known_report.append({
                "mode": mode, "index": i, "case": rows[i]["case"], "species": rows[i]["species"],
                "name": rows[i]["name"], "chip": chips[i],
                "input": metrics.check_stage(f"{mode}.input", inputs[i], ref_in[i], th["e2e"]["input"]),
                "embedding": metrics.check_stage(f"{mode}.embedding", embs[i], ref_emb[i], th["e2e"]["embedding"]),
                "cosine_distance": cos[i]})

        control = metrics.check_stage(f"{mode}.negative_control", embs[gated] * th["negative_control_scale"],
                                      ref_emb[gated], th["e2e"]["embedding"])
        controls.append({**control, "failed_as_required": not control["pass"]})

    passed = all(r["pass"] for r in results) and all(c["failed_as_required"] for c in controls)
    json.dump({
        "pass": passed,
        "thresholds_status": manifest["thresholds_status"],
        "results": results,
        "negative_controls": controls,
        "per_chip_cosine": per_chip,
        "known_divergence": {
            "excluded_rows": len(reported),
            "gated_e2e_rows": len(gated),
            "reason": sorted({k["reason"] for k in manifest["known_divergence"]}),
            "entries": manifest["known_divergence"],
            "reported": known_report,
        },
        "service_env": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0),
                        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                        "precision": m.precision},
        "reference_env": ref_meta["env"],
        "reference_source": ref_meta["reference_source"],
        "checkpoint_sha256": m.checkpoint_sha256,
    }, open(args.artifact, "w"), indent=2)
    print(json.dumps({"pass": passed, "thresholds_status": manifest["thresholds_status"]}))
    print(f"{len(reported)} of {len(rows)} rows excluded from gated e2e (reported only, see "
          f"'known_divergence' in the artifact): service chipper pads overhanging boxes, "
          f"training clamps; arch gated on all {len(rows)}.")
    code = metrics.exit_code(passed, manifest["thresholds_status"], args.allow_proposed)
    if code == 2:
        print(f"EXIT 2: checks passed but thresholds_status is {manifest['thresholds_status']!r}, "
              "not 'agreed'; this is not a release pass (use --allow-proposed to accept).")
    sys.exit(code)


if __name__ == "__main__":
    main()

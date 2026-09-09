#!/usr/bin/env python3
"""wbia-orientation fidelity gate — release-blocking host preflight.

Executes the pinned REFERENCE and the PORT live on the same bytes and compares
them to each other. There are no frozen expected values: they would pin one
implementation's output and rot on any dependency bump. See README.md.

Exit 0 = pass. Non-zero = do not deploy.
"""
import argparse, hashlib, json, math, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))


def circular_error(a: float, b: float) -> float:
    d = a - b
    return abs(math.atan2(math.sin(d), math.cos(d)))


def environment() -> dict:
    import numpy, torch, timm, skimage, imageio, PIL
    return {"torch": torch.__version__, "timm": timm.__version__,
            "numpy": numpy.__version__, "scikit-image": skimage.__version__,
            "imageio": imageio.__version__, "pillow": PIL.__version__}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fixtures", required=True, help="directory of fixture images")
    ap.add_argument("--artifact", default="preflight-artifact.json")
    args = ap.parse_args()

    from reference_runner import Reference
    from app.models.wbia_orientation import WbiaOrientationModel

    manifest = json.load(open(args.manifest))
    thr = manifest["thresholds"]
    env = environment()
    print(f"environment: {env}")

    rows, failures = [], []
    for ck in manifest["checkpoints"]:
        path = ck["path"]
        if not os.path.isfile(path):
            failures.append(f"checkpoint missing: {path}")
            continue
        digest = sha256(path)
        print(f"\n{ck['model_id']}  sha256={digest[:16]}...")
        ref = Reference(path)
        port = WbiaOrientationModel()
        port.load(model_id=ck["model_id"], checkpoint_path=path, device="cpu")

        for fx in manifest["fixtures"]:
            img = os.path.join(args.fixtures, fx.get("file", ""))
            if not os.path.isfile(img):
                failures.append(f"fixture missing: {img}")
                continue
            data = open(img, "rb").read()
            got = hashlib.sha256(data).hexdigest()
            if fx.get("sha256") not in (None, "", "<fixture byte hash>") and got != fx["sha256"]:
                failures.append(f"fixture {img} hash mismatch (manifest identity broken)")
                continue
            t_ref, c_ref = ref.theta(data, fx["bbox"])
            r = port.predict_batch(data, [fx["bbox"]])[0]
            e_t = circular_error(t_ref, r["theta"])
            e_c = max(abs(a - b) for a, b in zip(c_ref, r["coords_normalized"]))
            rows.append({"checkpoint": ck["model_id"], "fixture": fx.get("file"),
                         "stratum": fx.get("stratum"), "bbox": fx["bbox"],
                         "theta_ref": t_ref, "theta_port": r["theta"],
                         "theta_err": e_t, "coord_err": e_c,
                         "effective_bbox": r["effective_bbox"]})
            if e_t > thr["theta_circular_max_rad"]:
                failures.append(f"{ck['model_id']}/{fx.get('file')}: theta err {e_t:.3e}")
            if e_c > thr["coords_elementwise_max"]:
                failures.append(f"{ck['model_id']}/{fx.get('file')}: coord err {e_c:.3e}")

    # Enforce per-stratum minimums: a gate that silently ran 3 samples is not a gate.
    for name, spec in manifest.get("strata", {}).items():
        if name.startswith("_"):
            continue
        n = sum(1 for r in rows if r["stratum"] == name)
        if n < spec.get("min_samples", 0):
            failures.append(f"stratum '{name}': {n} samples < required {spec['min_samples']}")

    if rows:
        print(f"\nworst theta err: {max(r['theta_err'] for r in rows):.3e} "
              f"(<= {thr['theta_circular_max_rad']:.0e})")
        print(f"worst coord err: {max(r['coord_err'] for r in rows):.3e} "
              f"(<= {thr['coords_elementwise_max']:.0e})")

    json.dump({"environment": env, "results": rows, "failures": failures},
              open(args.artifact, "w"), indent=2)
    print(f"artifact -> {args.artifact}")

    if failures:
        print("\n*** GATE FAILED — DO NOT DEPLOY ***")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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


def wbia_localization_boundary(bbox, width, height):
    """What WBIA stored for a Wildbook lightnet detection, and so what the
    reference plugin later read via get_annot_bboxes: the `localizations` table
    clips every box to the frame (wbia/core_images.py:1539-1557, boundary=True).
    Transcribed independently of the port's resolve_bbox, so the gate compares
    the port's whole path (clip + slice) against WBIA's whole path."""
    xtl, ytl, w, h = (int(v) for v in bbox)
    xbr, ybr = xtl + w, ytl + h
    xtl, xbr = min(max(0, xtl), width), min(max(0, xbr), width)
    ytl, ybr = min(max(0, ytl), height), min(max(0, ybr), height)
    return [xtl, ytl, xbr - xtl, ybr - ytl]


def reference_image(data: bytes):
    """The image bytes WBIA would have handed the reference plugin, their
    upright (width, height), and the fixture's EXIF Orientation tag.

    WBIA rewrote EXIF-rotated images upright on import
    (wbia/algo/preproc/preproc_image.py:193-201: cv2.imread applies the tag,
    cv2.imwrite saves without it), so the plugin read upright pixels. For a
    tagged fixture, decode it the same way (cv2, independent of the port) and
    pass it on as a lossless PNG -- a frame-parity oracle; WBIA's own rewrite
    was a lossy JPEG. Only JPEG is supported for tagged fixtures: OpenCV 4.8
    ignores WebP EXIF, and IMREAD_COLOR would drop 16-bit depth. Untagged
    fixtures pass through byte for byte, as before.
    """
    import io
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        fmt, (w, h) = im.format, im.size
        orientation = im.getexif().get(0x0112)
    if orientation in (None, 0, 1):
        return data, w, h, orientation
    if fmt != "JPEG":
        raise ValueError(f"EXIF-tagged {fmt} fixture (Orientation={orientation}): the "
                         f"gate's EXIF parity oracle is only defined for JPEG")
    upright = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    ok, png = cv2.imencode(".png", upright)
    if not ok:
        raise RuntimeError("could not re-encode the upright fixture")
    h, w = upright.shape[:2]
    return png.tobytes(), w, h, orientation


def compare_fixture(model_id, fx, data, ref, port, thr, rows, failures):
    """Run one fixture through the reference and the port and record the result.
    The port gets the raw bytes and raw detection; the reference gets what WBIA
    would have stored: the upright image and the box its localization step
    clipped to that frame."""
    name = fx.get("file")
    ref_data, img_w, img_h, tag = reference_image(data)
    if fx.get("stratum") == "exif_rotated" and tag not in range(2, 9):
        failures.append(f"{model_id}/{name}: in exif_rotated but EXIF Orientation is {tag!r}")
        return
    ref_bbox = wbia_localization_boundary(fx["bbox"], img_w, img_h)
    t_ref, c_ref = ref.theta(ref_data, ref_bbox)
    r = port.predict_batch(data, [fx["bbox"]])[0]
    e_t = circular_error(t_ref, r["theta"])
    e_c = max(abs(a - b) for a, b in zip(c_ref, r["coords_normalized"]))
    rows.append({"checkpoint": model_id, "fixture": name,
                 "stratum": fx.get("stratum"), "bbox": fx["bbox"],
                 "reference_bbox": ref_bbox,
                 "theta_ref": t_ref, "theta_port": r["theta"],
                 "theta_err": e_t, "coord_err": e_c,
                 "effective_bbox": r["effective_bbox"]})
    if e_t > thr["theta_circular_max_rad"]:
        failures.append(f"{model_id}/{name}: theta err {e_t:.3e}")
    if e_c > thr["coords_elementwise_max"]:
        failures.append(f"{model_id}/{name}: coord err {e_c:.3e}")
    # Same crop, not just similar outputs: the port's effective_bbox must be the
    # box WBIA would have cropped (full frame when that is empty, animal_wbia.py:25-28).
    expected_eff = (ref_bbox if ref_bbox[2] > 0 and ref_bbox[3] > 0
                    else [0, 0, img_w, img_h])
    if r["effective_bbox"] != expected_eff:
        failures.append(f"{model_id}/{name}: effective_bbox "
                        f"{r['effective_bbox']} != WBIA crop {expected_eff}")


def stratum_shortfalls(rows, strata, checkpoints):
    """Per-stratum minimums, counted as DISTINCT fixtures per checkpoint: a gate
    that silently ran 3 samples (or one fixture three times) is not a gate."""
    out = []
    for name, spec in strata.items():
        if name.startswith("_"):
            continue
        need = spec.get("min_samples", 0)
        for ck in checkpoints or [None]:
            n = len({r["fixture"] for r in rows
                     if r["stratum"] == name and r["checkpoint"] == ck})
            if n < need:
                out.append(f"stratum '{name}' on {ck}: {n} fixtures < required {need}")
    return out


def resolve_reference_root(manifest, override=None):
    return override or manifest.get("reference_source", {}).get("path") or "/reference"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fixtures", required=True, help="directory of fixture images")
    ap.add_argument("--artifact", default="preflight-artifact.json")
    ap.add_argument("--reference-root", help="checkout root containing wbia_orientation")
    args = ap.parse_args(argv)

    from reference_runner import Reference
    from app.models.wbia_orientation import WbiaOrientationModel

    manifest = json.load(open(args.manifest))
    reference_root = resolve_reference_root(manifest, args.reference_root)
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
        ref = Reference(path, reference_root=reference_root)
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
            # Everything per fixture is recorded against it and the run goes on,
            # so one bad fixture cannot abort the gate before the artifact.
            try:
                compare_fixture(ck["model_id"], fx, data, ref, port, thr, rows, failures)
            except Exception as e:
                failures.append(f"{ck['model_id']}/{fx.get('file')}: {type(e).__name__}: {e}")

    # Enforce per-stratum minimums: a gate that silently ran 3 samples is not a gate.
    failures.extend(stratum_shortfalls(rows, manifest.get("strata", {}),
                                       [ck["model_id"] for ck in manifest["checkpoints"]]))

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

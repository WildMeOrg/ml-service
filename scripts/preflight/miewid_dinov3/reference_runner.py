"""Reference half of the MiewID DINOv3 parity gate. Runs in the TRAINING
venv (transformers 5.17, torch 2.6) with the training repo's own code:
load_image (cv2.imread, EXIF applied) -> get_chip_from_img ->
get_test_transforms((448, 448)) -> MiewIdNet, under CUDA fp16 autocast as
training eval runs.

For each bbox mode (router = int() of each coordinate, as /extract converts
it; float = the CSV values) and each chip it records the chip, the input
tensor and every stage:

  <out>/reference.npz   {mode}_input[N,3,448,448] {mode}_tokens[N,789,768]
                        {mode}_concat[N,1536] {mode}_embedding[N,1536]
                        {mode}_chip_{i} (uint8 HWC, sizes vary)
  <out>/reference.json  chip rows in order, image sha256s, environment,
                        checkpoint sha256, reference source + revision
  <out>/reference_chips.csv   the exact CSV used, for the service side

ASSUMPTIONS ABOUT THE TRAINING TREE (verify against the actual DINOv3
experiment tree before the first run; the public plugin tree does not have
the dinov3 branch): MiewIdNet accepts these keyword arguments, and
MiewIdNet(x) returns the BatchNorm output. If either is wrong, change the
two marked lines and record the change in --source-note.

Usage (on the training host):
  python reference_runner.py --repo /path/to/dinov3-exp --repo-revision <hash-or-tarball-md5> \
      --checkpoint model_final.bin --chips reference_chips.csv --path-column host_path \
      --out /tmp/dinov3-ref
"""
import argparse
import ast
import csv
import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np
import torch


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def router_bbox(bbox):
    return tuple(int(v) for v in bbox)   # == metrics.router_bbox; kept local (training venv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="tree containing wbia_miew_id/")
    ap.add_argument("--repo-revision", required=True, help="commit hash or source tarball md5")
    ap.add_argument("--source-note", default="", help="any deviation from the assumptions above")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--chips", required=True)
    ap.add_argument("--path-column", default="host_path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from wbia_miew_id.datasets.helpers import get_chip_from_img, load_image
    from wbia_miew_id.datasets.transforms import get_test_transforms
    from wbia_miew_id.models.model import MiewIdNet
    import transformers

    # ASSUMPTION 1: constructor keywords (see module docstring).
    model = MiewIdNet(n_classes=1, model_name="dinov3:facebook/dinov3-vitb16-pretrain-lvd1689m",
                      use_fc=False, pretrained=True)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval().cuda()
    assert model.backbone.config._attn_implementation == "sdpa", model.backbone.config._attn_implementation

    captured = {}

    def grab_tokens(mod, inp, out):
        captured["tokens"] = out.last_hidden_state.float()

    def grab_concat(mod, inp, out):
        captured["concat"] = inp[0].float()

    model.backbone.register_forward_hook(grab_tokens)
    model.bn.register_forward_hook(grab_concat)
    transform = get_test_transforms((448, 448))

    rows = list(csv.DictReader(open(args.chips)))
    arrays, meta = {}, []
    for mode in ("router", "float"):
        stacks = {"input": [], "tokens": [], "concat": [], "embedding": []}
        for i, row in enumerate(rows):
            path = row[args.path_column]
            bbox = ast.literal_eval(row["bbox"])
            bbox = list(router_bbox(bbox)) if mode == "router" else bbox
            img = load_image(path)
            chip = get_chip_from_img(img, bbox, float(row["theta"]))
            x = transform(image=chip)["image"].unsqueeze(0).cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                emb = model(x).float()   # ASSUMPTION 2: returns the BN output
            arrays[f"{mode}_chip_{i}"] = np.ascontiguousarray(chip, dtype=np.uint8)
            stacks["input"].append(x[0].cpu().numpy())
            stacks["tokens"].append(captured["tokens"][0].cpu().numpy())
            stacks["concat"].append(captured["concat"][0].cpu().numpy())
            stacks["embedding"].append(emb[0].cpu().numpy())
            if mode == "router":
                meta.append({"index": i, "case": row["case"], "species": row["species"],
                             "name": row["name"], "path": path, "bbox": row["bbox"],
                             "theta": row["theta"], "image_sha256": sha256_file(path)})
        for k, v in stacks.items():
            arrays[f"{mode}_{k}"] = np.stack(v)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "reference.npz", **arrays)
    shutil.copyfile(args.chips, out / "reference_chips.csv")
    json.dump({
        "chips": meta,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "reference_source": {"repo": args.repo, "revision": args.repo_revision,
                             "note": args.source_note},
        "env": {"torch": str(torch.__version__), "transformers": str(transformers.__version__),
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "attn_implementation": model.backbone.config._attn_implementation},
    }, open(out / "reference.json", "w"), indent=2)
    print(f"wrote {len(rows)} chips x 2 bbox modes to {out}")


if __name__ == "__main__":
    main()

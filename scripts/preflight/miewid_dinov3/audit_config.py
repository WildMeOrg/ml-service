"""Audit a production model_config.json for MiewID entries that would abort
service startup under this branch (app/main.py re-raises any load error).

  python3 -m scripts.preflight.miewid_dinov3.audit_config /path/to/model_config.json

Loads no weights and downloads nothing. Per model_type == "miewid" entry it
runs the same option validation and path resolution as MiewidModel.load, then
checks the `imgsz` / `embedding_dim` assertions against what that path implies.
Exit 1 if any entry would abort startup, else 0.
"""
import json
import sys

from app.models.miewid import (
    LOAD_PATH_DINOV3, LOAD_PATH_EFFNET, _parse_load_options, resolve_load_path,
)

DINOV3_SIZE, LEGACY_SIZE, DINOV3_DIM = 448, 440, 1536


def audit_entry(entry):
    """Return (problems, notes) for one miewid entry."""
    try:
        opts = _parse_load_options(entry)
    except ValueError as e:
        return [str(e)], []
    path = resolve_load_path(opts, entry.get("version", 3))
    problems, notes = [], []
    want = DINOV3_SIZE if path == LOAD_PATH_DINOV3 else LEGACY_SIZE
    if opts["imgsz"] is not None and opts["imgsz"] != want:
        problems.append(f"imgsz {opts['imgsz']} does not match the {path} path (expects {want})")
    dim = opts["embedding_dim"]
    if dim is not None:
        if path == LOAD_PATH_DINOV3:
            if dim != DINOV3_DIM:
                problems.append(f"embedding_dim {dim} != {DINOV3_DIM} for dinov3_vitb16")
        elif path == LOAD_PATH_EFFNET:
            notes.append(f"embedding_dim {dim}: verify against checkpoint (v5 = 2152)")
        else:
            problems.append(f"embedding_dim {dim} cannot be asserted on the {path} path")
    return problems, notes


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python3 -m scripts.preflight.miewid_dinov3.audit_config model_config.json")
        return 2
    models = json.load(open(argv[0], encoding="utf-8"))["models"]
    checked = failing = 0
    for entry in models:
        if entry.get("model_type") != "miewid":
            continue
        checked += 1
        problems, notes = audit_entry(entry)
        for r in problems:
            print(f"FAIL {entry.get('model_id')}: {r}")
        for r in notes:
            print(f"NOTE {entry.get('model_id')}: {r}")
        failing += bool(problems)
    print(f"summary: {checked} miewid entries checked, {failing} would abort startup")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())

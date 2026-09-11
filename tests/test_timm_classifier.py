"""Loader, preprocessing and label-semantics tests for TimmClassifierModel.

No ViT-L is ever built here: a tiny timm arch plus mocks covers every path.
The checkpoints are DeepFaune-shaped -- `args` + a `base_model.`-prefixed
`state_dict` + an unrelated pickled `transform` -- because that composition is
what the first real config entry exercises.
"""
import io
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
import timm
import torch
from PIL import Image

TINY = "vit_tiny_patch16_224"
_PATCH_CKPT = "app.models.timm_classifier.get_checkpoint_path"


def _tiny_state(num_classes=3, prefix="base_model.", global_pool="token"):
    m = timm.create_model(TINY, pretrained=False, num_classes=num_classes,
                          global_pool=global_pool)
    return {prefix + k: v for k, v in m.state_dict().items()}


def _ckpt(num_classes=3, prefix="base_model.", key="state_dict", args=True):
    ck = {key: _tiny_state(num_classes, prefix)}
    if args:
        ck["args"] = {"backbone": TINY, "num_classes": num_classes}
    ck["transform"] = "an unrelated pickled object"
    return ck


def _load(model=None, ckpt=None, **overrides):
    """Load a TimmClassifierModel against a synthetic checkpoint."""
    from app.models.timm_classifier import TimmClassifierModel
    cfg = dict(model_id="t", device="cpu", checkpoint_path="/fake/ck.pt",
               model_arch=TINY, img_size=224, global_pool="token",
               strip_prefix="base_model.", labels=["cat", "dog", "emu"])
    cfg.update(overrides)
    m = model or TimmClassifierModel()
    with patch("torch.load", return_value=ckpt if ckpt is not None else _ckpt()), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m.load(**cfg)
    return m


def _png_bytes(w=64, h=48, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue(), arr


# --- 1. the headline failure mode: global_pool must reach timm verbatim ------

def test_global_pool_is_passed_to_timm_create_model():
    real = timm.create_model
    with patch("app.models.timm_classifier.timm.create_model",
               side_effect=real) as spy:
        _load(global_pool="token")
    assert spy.call_args.kwargs["global_pool"] == "token"


def test_global_pool_none_is_not_forwarded():
    """None means 'use timm's default for this arch' -- don't pass the kwarg."""
    real = timm.create_model
    with patch("app.models.timm_classifier.timm.create_model",
               side_effect=lambda *a, **k: real(*a, **k)) as spy:
        _load(global_pool=None, ckpt=_ckpt())
    assert "global_pool" not in spy.call_args.kwargs


def test_get_model_info_surfaces_global_pool():
    m = _load(global_pool="token")
    assert m.get_model_info()["global_pool"] == "token"


# --- 2. unknown config keys are an error, not a silent fallback --------------

def test_misspelled_global_pool_raises_naming_the_key():
    with pytest.raises(ValueError, match="globl_pool"):
        _load(globl_pool="token")


def test_unknown_key_error_lists_every_offender():
    with pytest.raises(ValueError) as e:
        _load(nonsense=1, alsobad=2)
    assert "nonsense" in str(e.value) and "alsobad" in str(e.value)


def test_underscore_prefixed_keys_are_accepted_as_comments():
    m = _load(_note="mounted from prod registry")
    assert m.model_id == "t"


def test_model_path_is_accepted_and_ignored():
    """load_model always forwards model_path; it must not trip the check."""
    m = _load(model_path="")
    assert m.model_id == "t"


# --- 3. checkpoint must not be materialised on the GPU ----------------------

def test_torch_load_uses_cpu_map_location():
    with patch("torch.load", return_value=_ckpt()) as tl, \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        from app.models.timm_classifier import TimmClassifierModel
        TimmClassifierModel().load(
            model_id="t", device="cpu", checkpoint_path="/fake/ck.pt",
            model_arch=TINY, global_pool="token",
            strip_prefix="base_model.", labels=["cat", "dog", "emu"])
    assert tl.call_args.kwargs["map_location"] == "cpu"


# --- 4. checkpoint layouts and prefix stripping -----------------------------

@pytest.mark.parametrize("key", ["state_dict", "state", "model"])
def test_state_dict_extracted_from_each_known_wrapper_key(key):
    m = _load(ckpt=_ckpt(key=key))
    assert m.model is not None


def test_raw_state_dict_without_wrapper_key():
    m = _load(ckpt=_tiny_state(prefix=""), strip_prefix=None)
    assert m.model is not None


def test_raw_fallback_rejects_non_tensor_mapping():
    with pytest.raises(ValueError, match="state dict"):
        _load(ckpt={"epoch": 5, "notes": "nothing here"}, strip_prefix=None)


def test_missing_strip_prefix_surfaces_as_load_error():
    """Forgetting strip_prefix must fail loudly, not load a random subset."""
    with pytest.raises(RuntimeError):
        _load(strip_prefix=None)


# --- 5. class-count cross-checks --------------------------------------------

def test_checkpoint_args_num_classes_mismatch_raises():
    with pytest.raises(ValueError, match="num_classes"):
        _load(ckpt=_ckpt(num_classes=3), labels=["a", "b"])


def test_head_width_mismatch_raises_under_strict_load():
    ck = _ckpt(num_classes=3, args=False)
    with pytest.raises(RuntimeError):
        _load(ckpt=ck, labels=["a", "b", "c", "d"])


# --- 6. label validation -----------------------------------------------------

def test_neither_labels_nor_label_map_raises():
    with pytest.raises(ValueError, match="labels"):
        _load(labels=None)


def test_both_labels_and_label_map_raises():
    with pytest.raises(ValueError, match="exactly one"):
        _load(labels=["a", "b", "c"], label_map={0: "a", 1: "b", 2: "c"})


def test_label_map_with_sparse_keys_raises():
    with pytest.raises(ValueError, match="0"):
        _load(labels=None, label_map={0: "a", 1: "b", 5: "c"})


def test_label_map_coercion_collision_raises():
    with pytest.raises(ValueError, match="duplicate|collision"):
        _load(labels=None, label_map={"1": "a", "01": "b", "0": "c"})


def test_labels_as_bare_string_raises():
    with pytest.raises(ValueError, match="list"):
        _load(labels="cat")


def test_labels_with_non_string_value_raises():
    with pytest.raises(ValueError, match="string"):
        _load(labels=["cat", 7, "emu"])


def test_label_map_accepts_string_keys_zero_to_n():
    m = _load(labels=None, label_map={"0": "a", "1": "b", "2": "c"})
    assert m.label_map == {0: "a", 1: "b", 2: "c"}


# --- 7. label_mode semantics -------------------------------------------------

def _predict_once(m, seed=0):
    png, _ = _png_bytes(seed=seed)
    return m.predict(png)


def test_label_mode_species_emits_species_and_no_viewpoint():
    m = _load(label_mode="species")
    top = _predict_once(m)["predictions"][0]
    assert top["species"] == top["label"]
    assert top["viewpoint"] is None


def test_label_mode_viewpoint_emits_viewpoint_and_no_species():
    m = _load(label_mode="viewpoint", labels=["up", "down", "left"])
    top = _predict_once(m)["predictions"][0]
    assert top["viewpoint"] == top["label"]
    assert top["species"] is None


def test_label_mode_compound_delegates_to_parse_class_label():
    m = _load(label_mode="compound",
              labels=["zebra:left", "zebra:right", "giraffe:up"])
    top = _predict_once(m)["predictions"][0]
    assert top["species"] in {"zebra", "giraffe"}
    assert top["viewpoint"] in {"left", "right", "up"}


def test_label_mode_compound_honours_sentinel_prefixes():
    m = _load(label_mode="compound", sentinel_prefixes=["species"],
              labels=["species:left", "species:right", "species:up"])
    top = _predict_once(m)["predictions"][0]
    assert top["species"] is None
    assert top["viewpoint"] in {"left", "right", "up"}


def test_unknown_label_mode_raises():
    with pytest.raises(ValueError, match="label_mode"):
        _load(label_mode="nonsense")


# --- 8. preprocessing against an independent torchvision reference ----------

def _reference_tensor(arr, size=224, interp="bicubic", mean=None, std=None):
    from torchvision.transforms import InterpolationMode, transforms
    modes = {"bicubic": InterpolationMode.BICUBIC,
             "bilinear": InterpolationMode.BILINEAR}
    return transforms.Compose([
        transforms.Resize(size=(size, size), interpolation=modes[interp],
                          max_size=None, antialias=None),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean or [0.485, 0.456, 0.406]),
                             std=torch.tensor(std or [0.229, 0.224, 0.225])),
    ])(Image.fromarray(arr)).unsqueeze(0)


def test_preprocess_matches_torchvision_reference_bicubic():
    m = _load()
    png, arr = _png_bytes(seed=1)
    got = m._preprocess_image(png, None, 0.0)
    assert torch.allclose(got, _reference_tensor(arr), atol=1e-5)


def test_preprocess_honours_bilinear_interpolation():
    m = _load(interpolation="bilinear")
    png, arr = _png_bytes(seed=2)
    got = m._preprocess_image(png, None, 0.0)
    assert torch.allclose(got, _reference_tensor(arr, interp="bilinear"), atol=1e-5)
    assert not torch.allclose(got, _reference_tensor(arr), atol=1e-3)


def test_preprocess_honours_custom_mean_std():
    mean, std = [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]
    m = _load(mean=mean, std=std)
    png, arr = _png_bytes(seed=3)
    got = m._preprocess_image(png, None, 0.0)
    assert torch.allclose(got, _reference_tensor(arr, mean=mean, std=std), atol=1e-5)


def test_bbox_crop_without_square_expansion():
    m = _load(square_crop=False)
    png, arr = _png_bytes(w=64, h=48, seed=4)
    got = m._preprocess_image(png, (10, 8, 20, 10), 0.0)
    assert torch.allclose(got, _reference_tensor(arr[8:18, 10:30]), atol=1e-5)


def test_square_crop_expands_short_side_like_upstream():
    """DeepFaune cropSquareCVtoPIL: grow the short side by int(diff/2) each way."""
    m = _load(square_crop=True)
    png, arr = _png_bytes(w=64, h=48, seed=5)
    # bbox x=10 y=8 w=20 h=10 -> expand y by (20-10)//2 = 5 each way
    got = m._preprocess_image(png, (10, 8, 20, 10), 0.0)
    assert torch.allclose(got, _reference_tensor(arr[3:23, 10:30]), atol=1e-5)


def test_square_crop_clips_at_image_border():
    """Near an edge the crop stays a rectangle rather than running off."""
    m = _load(square_crop=True)
    png, arr = _png_bytes(w=64, h=48, seed=6)
    got = m._preprocess_image(png, (0, 0, 30, 10), 0.0)
    assert torch.allclose(got, _reference_tensor(arr[0:20, 0:30]), atol=1e-5)


def test_bbox_partially_out_of_frame_is_clipped_not_wrapped():
    m = _load(square_crop=False)
    png, arr = _png_bytes(w=64, h=48, seed=7)
    got = m._preprocess_image(png, (50, 40, 40, 40), 0.0)
    assert torch.allclose(got, _reference_tensor(arr[40:48, 50:64]), atol=1e-5)


def test_degenerate_bbox_raises():
    m = _load()
    png, _ = _png_bytes(seed=8)
    with pytest.raises(ValueError):
        m._preprocess_image(png, (10, 10, 0, 5), 0.0)


def test_theta_rotation_is_applied_after_the_crop():
    """Rotating the crop is not the same as cropping the rotated image."""
    m = _load()
    png, arr = _png_bytes(w=64, h=48, seed=9)
    rotated = m._preprocess_image(png, (10, 8, 20, 20), np.pi / 2)
    unrotated = m._preprocess_image(png, (10, 8, 20, 20), 0.0)
    assert not torch.allclose(rotated, unrotated, atol=1e-3)
    ref = _reference_tensor(
        np.asarray(Image.fromarray(arr[8:28, 10:30]).rotate(90.0)))
    assert torch.allclose(rotated, ref, atol=1e-5)


# --- 9. scalar config validation --------------------------------------------

@pytest.mark.parametrize("bad", [{"img_size": 0}, {"img_size": -8},
                                 {"interpolation": "nearest-ish"},
                                 {"mean": [0.5, 0.5]}, {"std": [1.0]},
                                 {"threshold": 1.5}, {"threshold": -0.1}])
def test_invalid_scalar_config_raises(bad):
    with pytest.raises(ValueError):
        _load(**bad)


# --- 10. output contract, registry and router wiring ------------------------

def test_predict_output_matches_efficientnet_contract():
    m = _load()
    out = _predict_once(m)
    assert set(out) >= {"model_id", "predictions", "all_probabilities",
                        "threshold", "bbox", "theta"}
    assert set(out["predictions"][0]) >= {"label", "index", "probability"}
    assert len(out["all_probabilities"]) == 3


def test_softmax_single_label_returns_exactly_one_prediction():
    m = _load(multi_label=False)
    out = _predict_once(m)
    assert len(out["predictions"]) == 1
    assert abs(sum(out["all_probabilities"]) - 1.0) < 1e-5


def test_multi_label_uses_sigmoid_and_threshold():
    m = _load(multi_label=True, threshold=0.0)
    out = _predict_once(m)
    assert len(out["predictions"]) == 3          # every class clears 0.0
    assert sum(out["all_probabilities"]) != pytest.approx(1.0, abs=1e-5)


def test_registry_exposes_timm_classifier():
    from app.models.model_handler import MODEL_REGISTRY
    assert MODEL_REGISTRY["timm-classifier"]["class"] == "TimmClassifierModel"


def test_pipeline_router_accepts_timm_classifier_in_classify_slot():
    import inspect
    from app.routers import pipeline_router
    from app.models.timm_classifier import TimmClassifierModel
    src = inspect.getsource(pipeline_router)
    assert "TimmClassifierModel" in src


# --- 11. end-to-end through ModelHandler into both endpoints ----------------

VALID_PNG_DATA_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _handler_with_real_timm_classifier(tmp_path):
    """A real TimmClassifierModel, loaded through ModelHandler from a
    DeepFaune-shaped checkpoint written to disk."""
    from app.models.model_handler import ModelHandler
    ck = tmp_path / "tiny-deepfaune.pt"
    torch.save(_ckpt(num_classes=3), ck)
    h = ModelHandler()
    h.load_model(model_id="df", model_type="timm-classifier", device="cpu",
                 checkpoint_path=str(ck), model_arch=TINY, img_size=224,
                 global_pool="token", strip_prefix="base_model.",
                 label_mode="species", labels=["red deer", "wolf", "bird"])
    return h


def test_end_to_end_classify_endpoint(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routers import classify_router

    h = _handler_with_real_timm_classifier(tmp_path)
    app = FastAPI()
    app.include_router(classify_router.router)
    app.state.model_handler = h
    r = TestClient(app).post("/classify/", json={
        "model_id": "df", "image_uri": VALID_PNG_DATA_URI})
    assert r.status_code == 200, r.text
    top = r.json()["predictions"][0]
    assert top["label"] in {"red deer", "wolf", "bird"}
    assert top["species"] == top["label"]


def test_end_to_end_pipeline_promotes_species_to_iaclass(tmp_path):
    """The contract that motivated label_mode: a bare species label must land
    on the result as top-level iaClass."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.models.lightnet_model import LightNetModel
    from app.models.miewid import MiewidModel
    from app.routers import pipeline_router

    h = _handler_with_real_timm_classifier(tmp_path)
    classifier = h.get_model("df")

    pm = MagicMock(spec=LightNetModel)
    pm.predict.return_value = {
        "model_id": "det", "bboxes": [[0, 0, 1, 1]], "scores": [0.9],
        "class_names": ["animal"], "num_detections": 1,
        "image_size": {"width": 1, "height": 1},
    }
    em = MagicMock(spec=MiewidModel)
    em.extract_embeddings.return_value = np.zeros((1, 8), dtype=np.float32)

    app = FastAPI()
    app.include_router(pipeline_router.router)
    handler = MagicMock()
    handler.get_model.side_effect = lambda mid: {
        "p": pm, "c": classifier, "e": em}.get(mid)
    handler.get_model_info.side_effect = lambda mid: {
        "p": {"config": {}}, "c": {"config": {}},
        "e": {"config": {"version": 4}}}.get(mid)
    handler.list_models.return_value = {"p": {}, "c": {}, "e": {}}
    app.state.model_handler = handler

    r = TestClient(app).post("/pipeline/", json={
        "image_uri": VALID_PNG_DATA_URI, "predict_model_id": "p",
        "classify_model_id": "c", "extract_model_id": "e",
        "bbox_score_threshold": 0.5})
    assert r.status_code == 200, r.text
    result = r.json()["results"][0]
    assert result["iaClass"] in {"red deer", "wolf", "bird"}
    assert "viewpoint" not in result

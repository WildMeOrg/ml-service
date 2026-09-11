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
    if ckpt is None and cfg["checkpoint_path"] != "/fake/ck.pt":
        # a real file on disk: let torch.load actually read it
        with patch(_PATCH_CKPT, side_effect=lambda p: p):
            m.load(**cfg)
        return m
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


def test_head_width_mismatch_raises():
    ck = _ckpt(num_classes=3, args=False)
    with pytest.raises(RuntimeError):
        _load(ckpt=ck, labels=["a", "b", "c", "d"])


def test_incomplete_state_dict_raises_only_because_load_is_strict():
    """A shape mismatch raises either way; a MISSING key is what proves
    strict=True. Flipping the implementation to strict=False passes the
    head-width test but fails this one."""
    state = _tiny_state(3)
    state.pop("base_model.norm.weight")
    with pytest.raises(RuntimeError, match="[Mm]issing"):
        _load(ckpt={"state_dict": state})


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
    with pytest.raises(ValueError, match="integer indices"):
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
    assert "viewpoint" not in top


def test_label_mode_viewpoint_emits_viewpoint_and_no_species():
    m = _load(label_mode="viewpoint", labels=["up", "down", "left"])
    top = _predict_once(m)["predictions"][0]
    assert top["viewpoint"] == top["label"]
    assert "species" not in top


def test_label_mode_compound_delegates_to_parse_class_label():
    """Not just 'produces a split' -- it must go through the shared helper,
    so compound semantics cannot drift between model types."""
    m = _load(label_mode="compound",
              labels=["zebra:left", "zebra:right", "giraffe:up"])
    with patch("app.models.timm_classifier.parse_class_label",
               return_value=("sentinel-species", "sentinel-viewpoint")) as spy:
        top = _predict_once(m)["predictions"][0]
    assert spy.called
    assert spy.call_args.kwargs["compound_labels"] is True
    assert top["species"] == "sentinel-species"
    assert top["viewpoint"] == "sentinel-viewpoint"


def test_label_mode_compound_honours_sentinel_prefixes():
    m = _load(label_mode="compound", sentinel_prefixes=["species"],
              labels=["species:left", "species:right", "species:up"])
    top = _predict_once(m)["predictions"][0]
    assert "species" not in top
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


def test_degenerate_bbox_raises_before_any_transform_runs():
    m = _load()
    png, _ = _png_bytes(seed=8)
    m.transforms = MagicMock(side_effect=AssertionError("transforms must not run"))
    with pytest.raises(ValueError, match="positive width and height"):
        m._preprocess_image(png, (10, 10, 0, 5), 0.0)
    m.transforms.assert_not_called()


def test_fully_out_of_frame_bbox_raises_even_with_square_crop():
    """Square expansion must not drag an off-image box back over the image:
    on 64x48, bbox (70, 4, 10, 40) expands x to 55..95 and would otherwise
    clip to 55..64 and classify a strip that was never requested."""
    m = _load(square_crop=True)
    png, _ = _png_bytes(w=64, h=48, seed=11)
    with pytest.raises(ValueError, match="does not overlap"):
        m._preprocess_image(png, (70, 4, 10, 40), 0.0)


def test_already_square_bbox_is_untouched_by_square_crop():
    m = _load(square_crop=True)
    png, arr = _png_bytes(w=64, h=48, seed=12)
    got = m._preprocess_image(png, (10, 8, 16, 16), 0.0)
    assert torch.allclose(got, _reference_tensor(arr[8:24, 10:26]), atol=1e-5)


def test_square_crop_with_odd_side_difference_floors_the_expansion():
    """int((w-h)/2) each side: an odd difference leaves a 1px shortfall,
    exactly as upstream does."""
    m = _load(square_crop=True)
    png, arr = _png_bytes(w=64, h=48, seed=13)
    # w=21 h=10 -> expand = int(11/2) = 5 -> y 3..23 (20 tall, not 21)
    got = m._preprocess_image(png, (10, 8, 21, 10), 0.0)
    assert torch.allclose(got, _reference_tensor(arr[3:23, 10:31]), atol=1e-5)


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

def _efficientnet_reference(**kw):
    """A real EfficientNetModel (tiny arch, empty state dict -- it loads
    non-strict) to compare response shapes against."""
    from app.models.efficientnet import EfficientNetModel
    kw.setdefault("label_map", {0: "cat", 1: "dog", 2: "emu"})
    e = EfficientNetModel()
    with patch("torch.load", return_value={}), \
         patch("app.models.efficientnet.get_checkpoint_path", side_effect=lambda p: p):
        e.load(model_id="e", device="cpu", checkpoint_path="/fake/e.pt",
               model_arch="efficientnet_b0", img_size=64, multi_label=False, **kw)
    return e


def test_top_level_keys_match_efficientnet_exactly():
    png, _ = _png_bytes(seed=20)
    mine = _load().predict(png)
    theirs = _efficientnet_reference().predict(png)
    assert set(mine) == set(theirs)


def test_prediction_entry_keys_match_efficientnet_in_each_label_mode():
    """EfficientNet omits species/viewpoint unless compound parsing is on.
    A consumer that distinguishes an absent key from a null one must see the
    same shape from both models."""
    png, _ = _png_bytes(seed=21)
    plain = set(_efficientnet_reference().predict(png)["predictions"][0])
    compound = set(_efficientnet_reference(
        parse_compound_labels=True,
        label_map={0: "cat:left", 1: "dog:right", 2: "emu:up"},
    ).predict(png)["predictions"][0])

    assert plain == {"label", "index", "probability"}
    assert compound == plain | {"species", "viewpoint"}

    species_mode = set(_load(label_mode="species").predict(png)["predictions"][0])
    viewpoint_mode = set(_load(label_mode="viewpoint").predict(png)["predictions"][0])
    compound_mode = set(_load(
        label_mode="compound",
        labels=["cat:left", "dog:right", "emu:up"]).predict(png)["predictions"][0])

    assert species_mode == plain | {"species"}
    assert viewpoint_mode == plain | {"viewpoint"}
    assert compound_mode == compound
    assert len(_load().predict(png)["all_probabilities"]) == 3


def test_softmax_single_label_returns_exactly_one_prediction():
    m = _load(multi_label=False)
    out = _predict_once(m)
    assert len(out["predictions"]) == 1
    assert abs(sum(out["all_probabilities"]) - 1.0) < 1e-5


def test_multi_label_applies_a_nontrivial_threshold():
    """Fixed logits -> sigmoid [0.881, 0.119, 0.622]. At threshold 0.5 exactly
    two classes qualify, so an implementation ignoring the threshold fails."""
    m = _load(multi_label=True, threshold=0.5)
    m.model = lambda t: torch.tensor([[2.0, -2.0, 0.5]])
    out = _predict_once(m)
    assert [p["index"] for p in out["predictions"]] == [0, 2]
    assert out["all_probabilities"] == pytest.approx([0.8808, 0.1192, 0.6225], abs=1e-3)


def test_softmax_mode_ignores_threshold_and_takes_argmax():
    m = _load(multi_label=False, threshold=0.99)
    m.model = lambda t: torch.tensor([[0.1, 3.0, 0.2]])
    out = _predict_once(m)
    assert [p["index"] for p in out["predictions"]] == [1]


def test_registry_exposes_timm_classifier():
    from app.models.model_handler import MODEL_REGISTRY
    assert MODEL_REGISTRY["timm-classifier"]["class"] == "TimmClassifierModel"


def test_pipeline_router_rejects_a_non_classifier_in_the_classify_slot():
    """Guards the allowlist from the other side: if the check were removed
    entirely, this would 200 instead of 400."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.models.miewid import MiewidModel
    from app.routers import pipeline_router

    wrong = MagicMock(spec=MiewidModel)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    handler = MagicMock()
    handler.get_model.side_effect = lambda mid: wrong
    handler.list_models.return_value = {"c": {}}
    app.state.model_handler = handler
    r = TestClient(app).post("/pipeline/", json={
        "image_uri": VALID_PNG_DATA_URI, "predict_model_id": "p",
        "classify_model_id": "c", "extract_model_id": "e"})
    assert r.status_code == 400


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


# --- 12. guards added after adversarial review ------------------------------

@pytest.mark.parametrize("bad", [
    {"multi_label": "false"}, {"square_crop": "false"},
    {"multi_label": 1}, {"square_crop": "true"},
    {"verify_checkpoint_transform": "no"},
])
def test_stringy_booleans_are_rejected(bad):
    """bool("false") is True -- a stringy config would silently swap sigmoid
    for softmax, or expand every crop."""
    with pytest.raises(ValueError, match="boolean"):
        _load(**bad)


@pytest.mark.parametrize("bad", [
    {"std": [0.0, 1.0, 1.0]}, {"std": [-0.2, 0.2, 0.2]},
    {"mean": [float("nan"), 0.0, 0.0]}, {"std": [float("inf"), 1.0, 1.0]},
])
def test_degenerate_normalization_is_rejected(bad):
    with pytest.raises(ValueError, match="finite|positive"):
        _load(**bad)


def test_nearest_interpolation_is_not_offered():
    with pytest.raises(ValueError, match="interpolation"):
        _load(interpolation="nearest")


@pytest.mark.parametrize("bad_key", [0.9, True, "01", " 1", "1.0"])
def test_non_canonical_label_map_keys_are_rejected(bad_key):
    """int(0.9) == 0 and int(True) == 1 would quietly reindex the map."""
    with pytest.raises(ValueError, match="integer indices|duplicate"):
        _load(labels=None, label_map={0: "a", bad_key: "b", 2: "c"})


def test_whitespace_only_label_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        _load(labels=["cat", "   ", "emu"])


def test_strip_prefix_matching_only_some_keys_is_rejected():
    """A prefix that matches part of the checkpoint means the prefix is wrong
    or the layout is mixed -- either way, not something to guess at."""
    state = _tiny_state(3)
    state["head_extra.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="does not match"):
        _load(ckpt={"state_dict": state})


def test_mixed_prefixed_and_bare_keys_are_rejected():
    """Bare + prefixed copies of one tensor would collapse last-writer-wins
    and still load strictly. Rejecting the mixed layout is the guard -- once
    every key must carry the prefix, a post-strip collision is impossible."""
    state = _tiny_state(3)
    state["head.weight"] = torch.zeros(3, 192)
    with pytest.raises(ValueError, match="does not match"):
        _load(ckpt={"state_dict": state}, strip_prefix="base_model.")


def _ckpt_with_transform(size=224, mean=None, std=None, num_classes=3):
    from torchvision.transforms import InterpolationMode, transforms
    ck = _ckpt(num_classes=num_classes)
    ck["transform"] = transforms.Compose([
        transforms.Resize(size=(size, size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean or [0.485, 0.456, 0.406]),
                             std=torch.tensor(std or [0.229, 0.224, 0.225])),
    ])
    return ck


def test_img_size_disagreeing_with_checkpoint_transform_raises():
    """Config alone cannot know 256 is wrong for a 224px checkpoint -- it
    loads strictly and merely predicts worse. The checkpoint knows."""
    with pytest.raises(ValueError, match="img_size"):
        _load(ckpt=_ckpt_with_transform(size=224), img_size=256)


def test_swapped_mean_and_std_are_caught_by_the_checkpoint_transform():
    with pytest.raises(ValueError, match="normalizes with"):
        _load(ckpt=_ckpt_with_transform(),
              mean=[0.229, 0.224, 0.225], std=[0.485, 0.456, 0.406])


def test_matching_checkpoint_transform_loads_cleanly():
    m = _load(ckpt=_ckpt_with_transform(), img_size=224)
    assert m.img_size == 224


def test_checkpoint_transform_check_can_be_disabled_deliberately():
    m = _load(ckpt=_ckpt_with_transform(size=224), img_size=256,
              verify_checkpoint_transform=False)
    assert m.img_size == 256


def test_checkpoint_without_transform_is_not_blocked():
    m = _load(ckpt=_ckpt(), img_size=256)
    assert m.img_size == 256


def test_checkpoint_sha256_mismatch_raises(tmp_path):
    ck = tmp_path / "ck.pt"
    torch.save(_ckpt(), ck)
    with pytest.raises(ValueError, match="digest mismatch"):
        _load(ckpt=None, checkpoint_path=str(ck), checkpoint_sha256="00" * 32)


def test_checkpoint_sha256_match_loads(tmp_path):
    import hashlib
    ck = tmp_path / "ck.pt"
    torch.save(_ckpt(), ck)
    digest = hashlib.sha256(ck.read_bytes()).hexdigest()
    from app.models.timm_classifier import TimmClassifierModel
    m = TimmClassifierModel()
    m.load(model_id="t", device="cpu", checkpoint_path=str(ck), model_arch=TINY,
           img_size=224, global_pool="token", strip_prefix="base_model.",
           labels=["cat", "dog", "emu"], checkpoint_sha256=digest)
    assert m.model is not None

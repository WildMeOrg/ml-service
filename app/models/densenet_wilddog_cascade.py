"""African wild dog body/tail labeler cascade.

Ports WBIA's three-model wild dog labeler (wbia/core_annots.py:2163-2233),
which is special-cased there for the single tag string
``wilddog_v3+wilddog_v2+wilddog_v1`` — every other multi-tag string raises
NotImplementedError in WBIA, so wild dog is the only species that uses it.

This is NOT an ensemble. The three models play distinct roles and vote:

  router (v3)     decides body vs tail, and supplies the tail class
  coat (v1)       supplies the body coat class
  viewpoint (v2)  supplies the body viewpoint

The router gates; the other two must agree with it or the annotation is
labelled ambiguous. ``wild_dog_ambiguous`` / ``wild_dog+tail_ambiguous``
exist only as this disagreement output — no individual model has an
"ambiguous" class — which is why ACW's IA.json declares them as iaClasses.

Why the roles are split this way is visible in the label spaces:
  v1 carries coat classes but only 3 viewpoints (left/other/right)
  v2 carries only plain wild_dog but 14 viewpoints (front, backleft, ...)
  v3 carries the 8 tail classes plus the wild_dog:ignore body gate
so the viewpoint is taken from v2 and the coat from v1 deliberately.

Each role is itself a 3-head DenseNetClassifierModel ensemble, so all
checkpoint loading, preprocessing and softmax-averaging is reused from
there rather than reimplemented.
"""
import logging
from typing import Any, Dict, List, Optional

from app.models.base_model import BaseModel
from app.models.densenet_classifier import DenseNetClassifierModel

logger = logging.getLogger(__name__)

# Body coat classes emitted by the coat (v1) model. Mirrors flag1 in
# core_annots.py:2188-2194 — order irrelevant, membership is the test.
BODY_COAT_CLASSES = frozenset({
    "wild_dog_dark",
    "wild_dog_general",
    "wild_dog_puppy",
    "wild_dog_standard",
    "wild_dog_tan",
})

# Species the viewpoint (v2) model emits for a body — flag2, line 2195.
VIEWPOINT_BODY_CLASSES = frozenset({"wild_dog", "wild_dog_puppy"})

# Species the router (v3) model emits for a body — flag3, line 2196.
ROUTER_BODY_CLASSES = frozenset({"wild_dog"})

# Labels each role must expose, used to fail fast on a mis-wired config.
#
# These are deliberately per-role *distinguishing* sets, not just "some body
# class": v2 and v3 BOTH emit the raw species `wild_dog`, so a species-level
# check cannot tell a router/viewpoint swap apart. The discriminators are:
#   - only v2 has rich body viewpoints (`wild_dog:front`)
#   - only v3 has the tail_triple_black / tail_double_* classes
#   - only v1 has the coat classes
# Each set also covers every class the decision table actually reads, so a
# truncated or partially-trained checkpoint is rejected rather than silently
# never firing a branch. Extra labels (e.g. v1/v2 tail classes, which the
# cascade ignores) are allowed — these are subset checks.
ROUTER_REQUIRED_LABELS = frozenset({
    "wild_dog:ignore",
    "wild_dog+tail_double_black_brown:ignore",
    "wild_dog+tail_double_black_white:ignore",
    "wild_dog+tail_general:ignore",
    "wild_dog+tail_long_black:ignore",
    "wild_dog+tail_long_white:ignore",
    "wild_dog+tail_short_black:ignore",
    "wild_dog+tail_standard:ignore",
    "wild_dog+tail_triple_black:ignore",
})

COAT_REQUIRED_LABELS = frozenset({
    "wild_dog_dark:left", "wild_dog_dark:other", "wild_dog_dark:right",
    "wild_dog_general:left", "wild_dog_general:other", "wild_dog_general:right",
    "wild_dog_standard:left", "wild_dog_standard:other", "wild_dog_standard:right",
    "wild_dog_tan:left", "wild_dog_tan:other", "wild_dog_tan:right",
    "wild_dog_puppy:ignore",
})

VIEWPOINT_REQUIRED_LABELS = frozenset({
    "wild_dog:back", "wild_dog:backleft", "wild_dog:backright",
    "wild_dog:down", "wild_dog:front", "wild_dog:frontleft",
    "wild_dog:frontright", "wild_dog:left", "wild_dog:right",
    "wild_dog:up", "wild_dog:upback", "wild_dog:upfront",
    "wild_dog:upleft", "wild_dog:upright",
    "wild_dog_puppy:ignore",
})

REQUIRED_LABELS = {
    "router": (ROUTER_REQUIRED_LABELS, "the v3 tail/body-gate model"),
    "coat": (COAT_REQUIRED_LABELS, "the v1 coat model"),
    "viewpoint": (VIEWPOINT_REQUIRED_LABELS, "the v2 viewpoint model"),
}

# Config keys accepted inside each role object. Anything else is a typo and
# would otherwise be silently swallowed by DenseNetClassifierModel.load(**kwargs).
ROLE_CONFIG_KEYS = frozenset({
    "checkpoint_path", "checkpoint_paths", "img_size", "compound_labels",
    "label_map", "sentinel_prefixes", "ensemble_indices",
})

AMBIGUOUS_BODY = "wild_dog_ambiguous"
AMBIGUOUS_TAIL = "wild_dog+tail_ambiguous"
AMBIGUOUS_VIEWPOINT = "ambiguous"
TAIL_VIEWPOINT = "ignore"

ROLES = ("router", "coat", "viewpoint")


def _is_compound_label(label: str) -> bool:
    """True iff `label` is exactly `<species>:<viewpoint>`, both halves set.

    parse_class_label() splits on the FIRST colon, so 'a:b:c' would silently
    yield viewpoint='b:c' and ':ignore' would yield species=''. Neither is a
    label this cascade's decision table can reason about.
    """
    parts = label.split(":")
    return len(parts) == 2 and all(p.strip() for p in parts)


class DenseNetWildDogCascadeModel(BaseModel):
    """Three-model role-based cascade for African wild dog annotations."""

    def __init__(self):
        super().__init__()
        self.members: Dict[str, DenseNetClassifierModel] = {}
        self.img_size: int = 224
        self.model_id: str = ""
        self.device: str = "cpu"

    def load(self, model_path: str = "", device: str = "cpu",
             model_id: str = "",
             router: Optional[Dict[str, Any]] = None,
             coat: Optional[Dict[str, Any]] = None,
             viewpoint: Optional[Dict[str, Any]] = None,
             img_size: int = 224,
             **kwargs) -> None:
        # Nothing is written to self until members are built AND validated:
        # a failed reload of an already-loaded instance must not leave the old
        # members paired with the new metadata.
        if kwargs:
            raise ValueError(
                f"densenet-wilddog-cascade '{model_id}': unknown config key(s) "
                f"{sorted(kwargs)}. Expected: router, coat, viewpoint, img_size."
            )

        specs = {"router": router, "coat": coat, "viewpoint": viewpoint}
        missing = [r for r in ROLES if not specs[r]]
        if missing:
            raise ValueError(
                f"densenet-wilddog-cascade '{model_id}' is missing required "
                f"role config(s): {missing}. Each of {list(ROLES)} must be an "
                f"object with checkpoint_path or checkpoint_paths."
            )

        # Build into a local dict so a failure part-way through cannot leave
        # this instance holding a half-wired cascade.
        members: Dict[str, DenseNetClassifierModel] = {}
        for role in ROLES:
            spec = dict(specs[role])
            unknown = set(spec) - ROLE_CONFIG_KEYS
            if unknown:
                raise ValueError(
                    f"densenet-wilddog-cascade '{model_id}': role '{role}' has "
                    f"unknown config key(s) {sorted(unknown)}. Expected any of: "
                    f"{sorted(ROLE_CONFIG_KEYS)}."
                )
            member = DenseNetClassifierModel()
            member.load(
                device=device,
                model_id=f"{model_id}[{role}]",
                checkpoint_path=spec.pop("checkpoint_path", None),
                checkpoint_paths=spec.pop("checkpoint_paths", None),
                img_size=spec.pop("img_size", img_size),
                # Every wild dog label is `<species>:<viewpoint>`; the cascade
                # reads the parsed species/viewpoint back off each member.
                compound_labels=spec.pop("compound_labels", True),
                **spec,
            )
            members[role] = member

        self._validate_roles(members, model_id)

        # Commit all state together, only once everything above succeeded.
        self.members = members
        self.model_id = model_id
        self.device = device
        self.img_size = img_size

        logger.info(
            f"Loaded DenseNetWildDogCascadeModel '{model_id}': "
            + ", ".join(
                f"{r}={len(self.members[r].label_map)} classes" for r in ROLES
            )
        )

    def _validate_roles(self, members: Dict[str, DenseNetClassifierModel],
                        model_id: str) -> None:
        """Fail fast if the checkpoint sets are wired to the wrong roles.

        The v1/v2/v3 archives look alike on disk and a swap would silently
        mislabel every annotation rather than error, so each role must expose
        the labels that distinguish its model AND every label the decision
        table reads. Checking raw species is NOT sufficient: v2 and v3 both
        emit species `wild_dog`, so a router/viewpoint swap would pass.
        """
        for role in ROLES:
            labels = set(members[role].label_map.values())

            # Members load with compound_labels=True, so parse_class_label()
            # must yield a usable (species, viewpoint) for every label. A
            # colon-free label yields species=None and would silently fail every
            # branch flag; an empty half or a second colon means the label isn't
            # what this cascade's decision table assumes. Reject at load rather
            # than at the first prediction.
            malformed = sorted(l for l in labels if not _is_compound_label(l))
            if malformed:
                raise ValueError(
                    f"densenet-wilddog-cascade '{model_id}': role '{role}' has "
                    f"non-compound label(s) {malformed}; every wild dog label "
                    f"must be exactly '<species>:<viewpoint>' with both halves "
                    f"non-empty."
                )

            required, description = REQUIRED_LABELS[role]
            absent = required - labels
            if absent:
                raise ValueError(
                    f"densenet-wilddog-cascade '{model_id}': role '{role}' "
                    f"is missing {len(absent)} label(s) required of "
                    f"{description}: {sorted(absent)[:5]}"
                    f"{' ...' if len(absent) > 5 else ''}. Its label space is "
                    f"{sorted(labels)}. Check that the correct checkpoints are "
                    f"assigned to each role."
                )

    def predict(self, image_bytes: bytes,
                bbox: Optional[List[int]] = None,
                theta: float = 0.0,
                **kwargs) -> Dict[str, Any]:
        tops = {}
        for role in ROLES:
            result = self.members[role].predict(
                image_bytes, bbox=bbox, theta=theta
            )
            tops[role] = result["predictions"][0]

        router_top = tops["router"]
        coat_top = tops["coat"]
        viewpoint_top = tops["viewpoint"]

        # flag1/flag2/flag3 in core_annots.py:2188-2196.
        coat_is_body = coat_top["species"] in BODY_COAT_CLASSES
        viewpoint_is_body = viewpoint_top["species"] in VIEWPOINT_BODY_CLASSES
        router_is_body = router_top["species"] in ROUTER_BODY_CLASSES

        # WBIA averages the three member confidences — np.mean, line 2198.
        score = (
            coat_top["probability"]
            + viewpoint_top["probability"]
            + router_top["probability"]
        ) / 3.0

        # Decision table — core_annots.py:2200-2217.
        if router_is_body:
            if coat_is_body and viewpoint_is_body:
                species = coat_top["species"]
                view = viewpoint_top["viewpoint"]
            else:
                species = AMBIGUOUS_BODY
                view = AMBIGUOUS_VIEWPOINT
        else:
            if not coat_is_body and not viewpoint_is_body:
                species = router_top["species"]
                view = TAIL_VIEWPOINT
            else:
                species = AMBIGUOUS_TAIL
                view = AMBIGUOUS_VIEWPOINT

        label = f"{species}:{view}"

        return {
            "model_id": self.model_id,
            "class": label,
            "probability": score,
            # The verdict is synthesized from three label spaces, so no single
            # member index describes it. -1 marks it as having no class index.
            "class_id": -1,
            "predictions": [{
                "label": label,
                "probability": score,
                "index": -1,
                "species": species,
                "viewpoint": view,
            }],
            # Per-member verdicts, for debugging a surprising cascade result.
            "cascade": {
                role: {
                    "label": tops[role]["label"],
                    "species": tops[role]["species"],
                    "viewpoint": tops[role]["viewpoint"],
                    "probability": tops[role]["probability"],
                }
                for role in ROLES
            },
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "densenet-wilddog-cascade",
            "device": str(self.device),
            "img_size": self.img_size,
            "members": {
                role: {
                    "num_classes": len(self.members[role].label_map),
                    "ensemble_size": len(self.members[role].models),
                    "label_map": self.members[role].label_map,
                }
                for role in ROLES
            },
        }

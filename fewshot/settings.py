"""Checks on a task's classifier settings and thresholds, and which of them belong to one embedder."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

from .classifiers import check_params

if TYPE_CHECKING:
    from .models import TaskConfig

THRESHOLDS = {"reject_below", "reject_below_zero_shot"}  # settings evaluation results don't depend on
# Settings added later, as read from a task saved before them: what keeps its old behaviour (new
# tasks get the field's default instead).
STORED_DEFAULTS = {"names_as_text": False}
MODEL_FITTED = ("temperature", "reject_below", "reject_below_zero_shot")  # on one embedder's score scale


def for_new_embedder(cfg: TaskConfig, keep: Collection[str] = ()) -> list[str]:
    """Clear, in place, the settings fitted to the previous embedder's similarities and
    confidences (``MODEL_FITTED`` not in ``keep``, and per-class thresholds), which mean
    something else under another model. Returns the names of those that had a value."""
    cleared = [k for k in MODEL_FITTED if k not in keep and getattr(cfg, k) is not None]
    for k in cleared:
        setattr(cfg, k, None)
    if any(c.reject_below is not None for c in cfg.classes):
        cfg.classes = [c.model_copy(update={"reject_below": None}) for c in cfg.classes]
        cleared.append("class thresholds")
    return cleared


# Settings that change a task's confidences, so a temperature fitted under the old ones is stale.
SCORING = ("classifier", "classifier_params", "text_weight", "names_as_text")


def for_new_scoring(cfg: TaskConfig, new: TaskConfig, keep: Collection[str] = ()) -> list[str]:
    """Clear, in place, ``cfg``'s fitted temperature (unless in ``keep``) if ``new`` changes a
    ``SCORING`` setting: it calibrated the old confidences. Returns ``["temperature"]`` if cleared."""
    if "temperature" in keep or cfg.temperature is None or all(getattr(cfg, k) == getattr(new, k) for k in SCORING):
        return []
    cfg.temperature = None
    return ["temperature"]


def check_settings(cfg: TaskConfig) -> None:
    """Raise ``ValueError`` unless ``cfg``'s classifier settings and thresholds are usable."""
    for key in ("reject_below", "reject_below_zero_shot"):
        v = getattr(cfg, key)
        if v is not None and not -1.0 <= v <= 1.0:
            raise ValueError(f"{key} must be between -1 and 1, got {v}")
    for c in cfg.classes:
        if c.reject_below is not None and not -1.0 <= c.reject_below <= 1.0:
            raise ValueError(f"The threshold of {c.name!r} must be between -1 and 1, got {c.reject_below}")
    if cfg.temperature is not None and not 0.01 <= cfg.temperature <= 100:
        raise ValueError(f"temperature must be between 0.01 and 100, got {cfg.temperature}")
    if "text_weight" in cfg.classifier_params:
        raise ValueError("text_weight is a task setting, not a classifier parameter")
    check_params(cfg.classifier, {"text_weight": cfg.text_weight, **cfg.classifier_params})

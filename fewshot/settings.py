"""Checks on a task's classifier settings and thresholds, and which of them belong to one embedder."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

from .classifiers import check_params

if TYPE_CHECKING:
    from .store import TaskConfig

THRESHOLDS = {"reject_below", "reject_below_zero_shot"}  # settings evaluation results don't depend on
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

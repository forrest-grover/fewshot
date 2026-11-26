"""Embedding-based few-shot image classification.

    from fewshot import Workspace, Pipeline

    ws = Workspace("./fewshot_data")
    task = ws.create("defects")
    task.add_example(open("ok1.jpg", "rb").read(), "ok")
    task.add_example(open("crack1.jpg", "rb").read(), "crack")
    pipe = Pipeline(ws)
    print(pipe.predict(task, ["new.jpg"])[0])
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Static view of the lazy exports below, for type checkers and linters.
    from .classifiers import CLASSIFIERS, ClassData, make_classifier
    from .embedders import DEFAULT_SPEC, PRESETS, get_embedder, register_embedder, unregister_embedder
    from .evaluate import benchmark, evaluate_task
    from .pipeline import Pipeline, Prediction
    from .store import Task, Workspace

# Lazy exports so `import fewshot.cli` doesn't pay for importing torch.
_EXPORTS = {
    "CLASSIFIERS": "classifiers", "ClassData": "classifiers", "make_classifier": "classifiers",
    "DEFAULT_SPEC": "embedders", "PRESETS": "embedders", "get_embedder": "embedders", "register_embedder": "embedders",
    "unregister_embedder": "embedders",
    "benchmark": "evaluate", "evaluate_task": "evaluate",
    "Pipeline": "pipeline", "Prediction": "pipeline",
    "Task": "store", "Workspace": "store",
}
__all__ = [
    "CLASSIFIERS",
    "DEFAULT_SPEC",
    "PRESETS",
    "ClassData",
    "Pipeline",
    "Prediction",
    "Task",
    "Workspace",
    "benchmark",
    "evaluate_task",
    "get_embedder",
    "make_classifier",
    "register_embedder",
    "unregister_embedder",
]


def __getattr__(name: str) -> Any:
    """Import the submodule that defines ``name`` on first access."""
    if name in _EXPORTS:
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """The module's names, the lazy exports among them (for ``dir()`` and completion)."""
    return sorted({*globals(), *__all__})

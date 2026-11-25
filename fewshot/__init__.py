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

# Lazy exports so `import fewshot.cli` doesn't pay for importing torch.
_EXPORTS = {
    "CLASSIFIERS": "classifiers", "ClassData": "classifiers", "make_classifier": "classifiers",
    "DEFAULT_SPEC": "embedders", "PRESETS": "embedders", "get_embedder": "embedders",
    "benchmark": "evaluate", "evaluate_task": "evaluate",
    "Pipeline": "pipeline", "Prediction": "pipeline",
    "Task": "store", "Workspace": "store",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(name)

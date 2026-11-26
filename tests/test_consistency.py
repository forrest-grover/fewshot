"""Results that must stay attached to the right task and model: late evaluations, the inbox of a
task with nothing to predict yet, per-class labels of rejected images, and switching embedder."""

import io

import pytest

from fewshot.archive import export_task, import_task
from fewshot.pipeline import Pipeline
from tests.conftest import png

RED, PINK, BLUE = (255, 0, 0), (200, 0, 60), (0, 0, 255)


def _task(ws, name="t"):
    t = ws.create(name, embedder="fake:color")
    for i in range(2):
        t.add_example(png(RED, noise=5, seed=i), "red")
        t.add_example(png(BLUE, noise=5, seed=i), "blue")
    return t


def test_late_evaluation_lands_on_its_own_task_only(ws):
    """An evaluation recorded after its task was deleted, or deleted and made again, is dropped."""
    old = _task(ws)
    report = {"revision": 7, "accuracy": 1.0, "n_examples": 4}
    old.record_evaluation(report)
    assert old.last_evaluation()["accuracy"] == 1.0
    ws.delete("t")
    old.record_evaluation(report)  # no error, nothing written
    new = ws.create("t", embedder="fake:color")
    old.record_evaluation(report)
    assert new.last_evaluation() is None


def test_inbox_lists_without_a_predictable_class(ws):
    """With classes but no examples or descriptions, the inbox lists its images unpredicted."""
    t = ws.create("t", embedder="fake:color")
    t.upsert_class("red")
    t.add_to_inbox([(png(RED), "a.png")])
    items = Pipeline(ws).inbox_suggestions(t)
    assert len(items) == 1
    assert "prediction" not in items[0]
    t.add_example(png(RED, noise=5), "red")
    assert "prediction" in Pipeline(ws).inbox_suggestions(t)[0]


def test_rejected_top_class_keeps_other_matching_labels(ws):
    """When the top class is under its own threshold, a class that clears its own is still listed."""
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    t.add_example(png(PINK), "pink")
    t.set_class_thresholds({"red": 0.9999, "pink": 0.5})
    p = Pipeline(ws).predict(t, [png((240, 0, 30))])[0]
    assert (p.label, p.top_label, p.rejected, p.labels) == (None, "red", True, ["pink"])


def test_switching_embedder_clears_what_was_fitted_to_the_old_one(ws):
    """Temperature and thresholds (task and per class) are cleared unless given too; undo restores."""
    t = _task(ws)
    t.update_settings(temperature=2.0, reject_below=0.5, reject_below_zero_shot=0.2)
    t.set_class_thresholds({"red": 0.8})
    assert t.update_settings(classifier="knn") == []
    assert t.update_settings(embedder="fake:color") == []  # the same model: nothing to clear
    cleared = t.update_settings(embedder="fake:other", reject_below=0.6)
    assert cleared == ["temperature", "reject_below_zero_shot", "class thresholds"]
    cfg = t.config
    assert (cfg.temperature, cfg.reject_below, cfg.reject_below_zero_shot) == (None, 0.6, None)
    assert all(c.reject_below is None for c in cfg.classes)
    t.undo()
    assert (t.config.embedder, t.config.temperature, t.config.classes[0].reject_below) == ("fake:color", 2.0, 0.8)


@pytest.mark.parametrize("embedder", [None, "fake:color", "fake:other"])
def test_import_with_another_embedder_clears_fitted_settings(ws, embedder):
    """Importing under another embedder drops the old model's temperature and thresholds."""
    t = _task(ws)
    t.update_settings(temperature=2.0)
    t.set_class_thresholds({"red": 0.8})
    buf = io.BytesIO()
    export_task(ws, "t", buf)
    buf.seek(0)
    cfg = import_task(ws, buf, name="copy", embedder=embedder).config
    fitted = (cfg.temperature, cfg.classes[0].reject_below)
    assert fitted == ((None, None) if embedder == "fake:other" else (2.0, 0.8))


def test_threshold_changes_keep_the_evaluation_current(ws):
    """Evaluation doesn't depend on reject thresholds: changing them keeps a current evaluation
    current, while other changes, or an evaluation already out of date, are left alone."""
    t = _task(ws)
    t.record_evaluation({"revision": t.config.revision, "accuracy": 1.0})
    t.set_class_thresholds({"red": 0.8})
    t.update_settings(reject_below=0.5, reject_below_zero_shot=0.1)
    assert t.last_evaluation()["revision"] == t.config.revision
    t.update_settings(classifier="knn")
    old = t.last_evaluation()["revision"]
    assert old != t.config.revision
    t.set_class_thresholds({"red": 0.7})
    assert t.last_evaluation()["revision"] == old

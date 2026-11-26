"""Task settings: the rules for changing them, and what a change clears or keeps."""

import io
import json
import zipfile

import pytest

from fewshot import services
from fewshot.archive import export_task, import_task
from fewshot.models import TaskConfig
from fewshot.store import Workspace
from tests.conftest import red_blue_task, two_each


# ---- rules ---------------------------------------------------------------
def test_settings_rules_live_in_the_library(ws):
    """Param merging, a new classifier's fresh params and temperature 1 are library rules."""
    t = two_each(ws)
    t.update_settings(classifier="knn", classifier_params={"k": 3})
    assert t.config.classifier_params == {"k": 3}
    t.update_settings(merge_params={"scale": 20})
    assert t.config.classifier_params == {"k": 3, "scale": 20}  # merged
    t.update_settings(classifier="prototype")
    assert t.config.classifier_params == {}  # a new classifier starts afresh
    t.update_settings(classifier="knn", merge_params={"k": 2})
    assert t.config.classifier_params == {"k": 2}
    t.update_settings(temperature=2.0)
    t.update_settings(temperature=1)
    assert t.config.temperature is None  # 1 means uncalibrated, from any interface


def test_apply_settings_checks_clears_and_embedders(ws):
    """apply_settings refuses a setting both set and cleared, naming the setting (not any
    interface's option), and checks only a new embedder."""
    t = two_each(ws)
    t.update_settings(reject_below=0.5)
    with pytest.raises(services.SettingConflict, match="reject_below can't be both set and cleared") as e:
        services.apply_settings(t, services.SettingsChange({"reject_below": 0.4}, frozenset({"reject_below"})))
    assert e.value.setting == "reject_below"
    services.apply_settings(t, services.SettingsChange(clear=frozenset({"reject_below"})))
    assert t.config.reject_below is None
    checked: list[str] = []
    services.apply_settings(t, services.SettingsChange({"embedder": "fake:color"}), embedder_ok=checked.append)
    assert not checked  # unchanged: not checked again
    with pytest.raises(ValueError, match="nobackend"):
        services.apply_settings(t, services.SettingsChange({"embedder": "nobackend:x"}))
    services.apply_settings(t, services.SettingsChange({"classifier": "knn"}, merge_params={"k": 2}))
    assert t.config.classifier_params == {"k": 2}
    for bad in (-5, 101, float("nan")):
        with pytest.raises(ValueError, match=r"^text_weight must be between 0 and 100"):
            services.apply_settings(t, services.SettingsChange({"text_weight": bad}))


def test_change_settings_names_the_request_fields(ws):
    """change_settings, for the REST API, says which request fields clash."""
    t = two_each(ws)
    with pytest.raises(ValueError, match="Give either reject_below or clear_reject, not both"):
        services.change_settings(t, {"reject_below": 0.4}, {"reject_below": "clear_reject"})
    services.change_settings(t, {"reject_below": 0.4})
    assert t.config.reject_below == 0.4


def test_check_specs():
    """Specs are stripped and deduplicated, and their number checked."""
    assert services.check_specs([" a ", "b", "a", ""]) == ["a", "b"]
    with pytest.raises(ValueError, match="at least one"):
        services.check_specs([" "])
    with pytest.raises(ValueError, match="between 1 and 1"):
        services.check_specs(["a", "b"], limit=1)


def test_zero_shot_threshold_and_flip_settings(ws):
    """The zero-shot threshold is range-checked like reject_below; flip is a plain switch."""
    t = ws.create("t", embedder="fake:color")
    t.update_settings(reject_below_zero_shot=0.25, flip=True)
    assert (t.config.reject_below_zero_shot, t.config.flip) == (0.25, True)
    with pytest.raises(ValueError, match="reject_below_zero_shot must be between"):
        t.update_settings(reject_below_zero_shot=2)
    assert t.summary()["revision"] == t.config.revision


# ---- what a change clears ------------------------------------------------
def test_switching_embedder_clears_what_was_fitted_to_the_old_one(ws):
    """Temperature and thresholds (task and per class) are cleared unless given too; undo restores."""
    t = red_blue_task(ws)
    t.update_settings(temperature=2.0, reject_below=0.5, reject_below_zero_shot=0.2)
    t.set_class_thresholds({"red": 0.8})
    assert t.update_settings(classifier="knn", temperature=2.0) == []  # kept: given too
    assert t.update_settings(embedder="fake:color") == []  # the same model: nothing to clear
    cleared = t.update_settings(embedder="fake:other", reject_below=0.6)
    assert cleared == ["temperature", "reject_below_zero_shot", "class thresholds"]
    cfg = t.config
    assert (cfg.temperature, cfg.reject_below, cfg.reject_below_zero_shot) == (None, 0.6, None)
    assert all(c.reject_below is None for c in cfg.classes)
    t.undo()
    assert (t.config.embedder, t.config.temperature, t.config.classes[0].reject_below) == ("fake:color", 2.0, 0.8)


@pytest.mark.parametrize("change", [{"classifier": "knn"}, {"classifier": "knn", "classifier_params": {"k": 1}},
                                    {"text_weight": 0.5}, {"names_as_text": False}])
def test_changing_how_scores_are_made_clears_the_temperature(ws, change):
    """A temperature calibrates one classifier's confidences: changing the classifier, its
    parameters, the text weight or names-as-text clears it (thresholds stay); undo restores it."""
    t = red_blue_task(ws)
    t.update_settings(temperature=2.0, reject_below=0.5)
    assert t.update_settings(**change) == ["temperature"]
    assert (t.config.temperature, t.config.reject_below) == (None, 0.5)
    t.undo()
    assert t.config.temperature == 2.0
    assert t.update_settings(flip=True) == []  # not a scoring setting
    assert t.config.temperature == 2.0


def test_changing_no_setting_saves_nothing(ws):
    """Settings given as they already are (or none) make no revision and no undo step."""
    t = red_blue_task(ws)
    rev, label = t.config.revision, t.undo_label()
    assert t.update_settings() == []
    assert t.update_settings(classifier=t.config.classifier, flip=False) == []
    assert (t.config.revision, t.undo_label()) == (rev, label)


def test_threshold_changes_keep_the_evaluation_current(ws):
    """Evaluation doesn't depend on reject thresholds: changing them keeps a current evaluation
    current, while other changes, or an evaluation already out of date, are left alone."""
    t = red_blue_task(ws)
    t.record_evaluation({"revision": t.config.revision, "accuracy": 1.0})
    t.set_class_thresholds({"red": 0.8})
    t.update_settings(reject_below=0.5, reject_below_zero_shot=0.1)
    assert t.last_evaluation()["revision"] == t.config.revision
    t.update_settings(classifier="knn")
    old = t.last_evaluation()["revision"]
    assert old != t.config.revision
    t.set_class_thresholds({"red": 0.7})
    assert t.last_evaluation()["revision"] == old


def _without_names_as_text(f):
    data = json.loads(f.read_bytes())
    del data["names_as_text"]
    f.unlink()  # kept versions are hard links to replaced task.json files
    f.write_text(json.dumps(data))


def test_tasks_saved_before_names_as_text_keep_it_off(ws):
    """A task.json, kept version or export from before names_as_text reads it as off, so older
    tasks predict as they did; new tasks have it on."""
    t = red_blue_task(ws)
    assert t.config.names_as_text is True
    t.upsert_class("x")
    for f in [t.path / "task.json", *(t.path / "history").glob("*.json")]:
        _without_names_as_text(f)
    old = Workspace(ws.root).get("t")
    assert old.config.names_as_text is False
    assert old.version(old.config.revision - 1).names_as_text is False
    old.update_settings(names_as_text=True)
    buf, older = io.BytesIO(), io.BytesIO()
    export_task(ws, "t", buf)
    with zipfile.ZipFile(buf) as src, zipfile.ZipFile(older, "w") as dst:
        for info in src.infolist():
            data = json.loads(src.read(info)) if info.filename == "task.json" else None
            if data is not None:
                del data["names_as_text"]
            dst.writestr(info, src.read(info) if data is None else json.dumps(data))
    assert import_task(ws, older, name="copy").config.names_as_text is False
    old.undo()
    old.undo()
    assert old.config.names_as_text is False
    assert TaskConfig(name="new").names_as_text is True
    assert TaskConfig.model_validate({"name": "new"}).names_as_text is True

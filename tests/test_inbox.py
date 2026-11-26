"""The inbox: unlabelled images, the model's suggestions, accepting and dismissing them."""

import importlib
import io
import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import fewshot.api
from fewshot import cli, cli_common
from fewshot.archive import export_task, import_task
from fewshot.pipeline import Pipeline
from fewshot.store import Workspace
from tests.conftest import png

RED, BLUE = (220, 30, 30), (30, 30, 220)


def _task(ws):
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    t.add_example(png(BLUE), "blue")
    return t


def test_add_accept_dismiss_and_undo(ws):
    """Images wait in the inbox, are labelled into examples or dismissed, each one undo step."""
    t = _task(ws)
    out = t.add_to_inbox([(png(RED, noise=10, seed=i), f"r{i}.png") for i in range(3)]
                         + [(b"junk", "x"), (png(RED), "dup.png")])
    assert [type(r).__name__ for r in out] == ["InboxItem"] * 3 + ["ValueError", "ValueError"]
    assert "Already an example of red" in str(out[4])
    assert t.undo_label() == "add 3 images to the inbox"
    again = t.add_to_inbox([(png(RED, noise=10, seed=0), "again.png")])
    assert again[0].id == out[0].id  # already waiting: no duplicate
    assert len(t.config.inbox) == 3
    assert t.image_path(out[0].id).exists()
    ids = [x.id for x in t.config.inbox]
    assert t.accept({ids[0]: "red", ids[1]: "pink"}) == 2
    assert t.undo_label() == "label 2 images from the inbox as pink, red"
    assert {e.label for e in t.config.examples if e.id in ids} == {"red", "pink"}
    assert "pink" in t.class_names()
    assert [x.id for x in t.config.inbox] == [ids[2]]
    t.undo()
    assert len(t.config.inbox) == 3
    assert t.dismiss([ids[2]]) == 1
    assert (t.path / out[2].file).exists()  # kept for undo
    t.undo()
    assert len(t.config.inbox) == 3
    with pytest.raises(KeyError, match="No inbox image 'nope'"):
        t.accept({"nope": "red"})
    with pytest.raises(ValueError, match="Class name is empty"):
        t.accept({ids[0]: " "})
    with pytest.raises(KeyError):
        t.dismiss(["nope"])
    assert len(t.config.inbox) == 3  # nothing changed by the failures


def test_adding_an_example_takes_it_out_of_the_inbox(ws):
    """An image added as an example directly is no longer waiting."""
    t = _task(ws)
    t.add_to_inbox([(png(RED, noise=10, seed=5), "a.png")])
    t.add_example(png(RED, noise=10, seed=5), "red")
    assert t.config.inbox == []
    assert t.summary()["n_inbox"] == 0


def test_inbox_survives_export_import_and_gc(ws, tmp_path):
    """Inbox images are exported and imported, and gc keeps their files."""
    t = _task(ws)
    t.add_to_inbox([(png(RED, noise=10, seed=7), "a.png")])
    assert ws.gc()["images"] == 0
    assert t.image_path(t.config.inbox[0].id).exists()
    buf = io.BytesIO()
    export_task(ws, "t", buf)
    buf.seek(0)
    u = import_task(Workspace(tmp_path / "other"), buf)
    assert [x.source for x in u.config.inbox] == ["a.png"]
    assert u.image_path(u.config.inbox[0].id).read_bytes() == png(RED, noise=10, seed=7)


def test_suggestions_and_their_order(ws):
    """Each inbox image gets a suggestion; the sorts put unsure, confident, novel or old first."""
    t = _task(ws)
    pipe = Pipeline(ws)
    assert pipe.inbox_suggestions(t) == []
    t.add_to_inbox([(png(RED, noise=5), "red.png"), (png((128, 30, 128)), "purple.png"),
                    (png((128, 128, 128)), "grey.png")])
    added = [i["source"] for i in pipe.inbox_suggestions(t, "added")]
    assert added == ["red.png", "purple.png", "grey.png"]
    confident = pipe.inbox_suggestions(t, "confident")
    assert confident[0]["source"] == "red.png"
    assert confident[0]["prediction"]["label"] == "red"
    assert len(confident[0]["prediction"]["scores"]) == 2
    assert pipe.inbox_suggestions(t, "uncertain")[0]["source"] == "purple.png"
    assert pipe.inbox_suggestions(t, "novel")[0]["source"] != "red.png"
    with pytest.raises(ValueError, match="sort must be one of"):
        pipe.inbox_suggestions(t, "random")
    bare = ws.create("bare", embedder="fake:color")
    bare.add_to_inbox([(png(RED), "a.png")])
    assert "prediction" not in pipe.inbox_suggestions(bare)[0]  # no classes yet


# ---- API ------------------------------------------------------------------
@pytest.fixture(name="client")
def client_fixture(tmp_path, monkeypatch):
    """A TestClient on a fresh API with task 't' (one red, one blue example)."""
    monkeypatch.setenv("FEWSHOT_HOME", str(tmp_path / "ws"))
    c = TestClient(importlib.reload(fewshot.api).app)
    c.post("/api/tasks", json={"name": "t", "embedder": "fake:color"})
    for label, rgb in (("red", RED), ("blue", BLUE)):
        c.post("/api/tasks/t/examples", data={"label": label}, files=[("files", ("x.png", png(rgb), "image/png"))])
    return c


def test_api(client, monkeypatch):
    """Upload, list with suggestions, accept and dismiss over the API."""
    files = [("files", (f"{i}.png", png(RED, noise=10, seed=i), "image/png")) for i in range(3)]
    r = client.post("/api/tasks/t/inbox", files=[*files, ("files", ("bad.png", b"junk", "image/png"))])
    assert r.status_code == 201
    assert (len(r.json()["added"]), len(r.json()["errors"]), r.json()["task"]["inbox_count"]) == (3, 1, 3)
    r = client.post("/api/tasks/t/inbox", files=[("files", ("bad.png", b"junk", "image/png"))])
    assert r.status_code == 400
    page = client.get("/api/tasks/t/inbox", params={"sort": "confident", "limit": 2}).json()
    assert (page["total"], len(page["items"])) == (3, 2)
    assert page["items"][0]["prediction"]["label"] == "red"
    assert client.get("/api/tasks/t/inbox", params={"sort": "nope"}).status_code == 422
    ids = [i["id"] for i in client.get("/api/tasks/t/inbox").json()["items"]]
    assert client.get(f"/api/tasks/t/examples/{ids[0]}/image").status_code == 200
    t = client.post("/api/tasks/t/inbox/accept", json={"labels": {ids[0]: "red"}}).json()
    assert (t["inbox_count"], len(t["examples"])) == (2, 3)
    assert client.post("/api/tasks/t/inbox/accept", json={"labels": {"nope": "red"}}).status_code == 404
    assert client.post("/api/tasks/t/inbox/accept", json={"labels": {ids[1]: "a/b"}}).status_code == 400
    assert client.post("/api/tasks/t/inbox/dismiss", json={"ids": [ids[1]]}).json()["inbox_count"] == 1
    assert client.post("/api/tasks/t/inbox/dismiss", json={"ids": ["nope"]}).status_code == 404

    monkeypatch.setattr(fewshot.api, "ALLOW_URLS", True)
    monkeypatch.setattr(fewshot.api, "fetch_url", lambda url, *a, **k: png(BLUE, noise=10, seed=9))
    r = client.post("/api/tasks/t/inbox/urls", json={"urls": ["http://h/b.png"]})
    assert r.json()["added"][0]["source"] == "http://h/b.png"

    def broken(*_args):
        raise ValueError("Could not load embedder")

    monkeypatch.setattr(fewshot.api.pipe, "inbox_suggestions", broken)
    assert client.get("/api/tasks/t/inbox").status_code == 400


# ---- CLI ------------------------------------------------------------------
def test_cli(tmp_path):
    """inbox add/ls/accept/dismiss from the command line."""
    root = tmp_path / "ws"
    _task(Workspace(root))
    runner = CliRunner()

    def run(*args):
        return runner.invoke(cli.app, [str(a) for a in args] + ["--root", str(root)])

    d = tmp_path / "new"
    d.mkdir()
    for i in range(3):
        (d / f"{i}.png").write_bytes(png(RED, noise=10, seed=i))
    (d / "bad.png").write_bytes(b"junk")
    r = run("inbox", "add", "t", d)
    assert "3 images in the inbox" in r.output
    assert "skip" in r.output
    assert run("inbox", "add", "t", d / "bad.png").exit_code == 1
    lines = run("inbox", "ls", "t", "--sort", "confident").output.splitlines()
    assert len(lines) == 4
    assert "red" in lines[0]
    items = json.loads(run("inbox", "ls", "t", "--json").output)
    first = items[0]["id"][:8]
    assert run("inbox", "accept", "t", first, "--label", "blue").output.strip() == "Labelled 1 inbox images; 2 left"
    assert run("inbox", "accept", "t").exit_code == 1  # neither --label nor --min-confidence
    assert run("inbox", "accept", "t", "zzzz", "--label", "x").exit_code == 1
    assert run("inbox", "accept", "t", "--min-confidence", "1.0").output.strip() == "Nothing to accept"
    assert "Labelled 2 inbox images; 0 left" in run("inbox", "accept", "t", "--min-confidence", "0.5").output
    run("inbox", "add", "t", tmp_path / "new" / "0.png")  # labelled already: refused
    run("inbox", "add", "t", tmp_path / "x.png")
    (tmp_path / "y.png").write_bytes(png(BLUE, noise=10, seed=3))
    run("inbox", "add", "t", tmp_path / "y.png")
    item = Workspace(root).get("t").config.inbox[0].id
    assert run("inbox", "dismiss", "t", item[:6]).output.strip() == "Dismissed 1 images; 0 left"
    assert run("inbox", "ls", "t", "--sort", "nope").exit_code == 1


def test_inbox_edges(ws, monkeypatch):
    """Chunked adds are one undo step; a dismissed image added again reuses its kept file; example
    lookups still work with images waiting; CLI errors for URLs and bad labels."""
    t = _task(ws)
    t.add_to_inbox([(png(RED, noise=10, seed=i), f"{i}.png") for i in range(3)], chunk=2)
    assert len(t.config.inbox) == 3
    t.undo()
    assert t.config.inbox == []
    t.redo()
    first = t.config.inbox[0]
    t.dismiss([first.id])
    t.add_to_inbox([(png(RED, noise=10, seed=0), "back.png")])  # its file was kept for undo
    assert t.image_path(first.id).exists()
    assert t.image_path(t.config.examples[0].id).exists()  # an example, with images waiting

    def no_fetch(url, *_a, **_k):
        raise ValueError(f"Could not fetch {url}")

    monkeypatch.setattr(cli_common, "fetch_url", no_fetch)
    runner = CliRunner()
    r = runner.invoke(cli.app, ["inbox", "add", "t", "http://h/x.png", "--root", str(ws.root)])
    assert "skip http://h/x.png: Could not fetch" in r.output
    r = runner.invoke(cli.app, ["inbox", "accept", "t", first.id, "--label", "a/b", "--root", str(ws.root)])
    assert r.exit_code == 1, r.output
    assert "can't be" in r.output

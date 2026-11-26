"""Classification and evaluation features: flip averaging, prompt ensembles, multi-label and
one-class prediction, near-duplicates, fast leave-one-out, metrics, devices, OOM and URL fetching."""

import http.server
import socket
import threading

import numpy as np
import pytest
import torch
from PIL import Image

import fewshot.evaluate as ev
from fewshot import embedders, fetch
from fewshot.classifiers import ClassData, top_k
from fewshot.embedders import Embedder, register_embedder, unregister_embedder
from fewshot.evaluate import evaluate_task
from fewshot.pipeline import FLIP_KEY, Pipeline
from fewshot.store import TaskConfig
from tests.conftest import ColorEmbedder, png

RED, GREEN, BLUE = (220, 30, 30), (30, 220, 30), (30, 30, 220)


class LeftRight(Embedder):  # pylint: disable=abstract-method
    """Mean colour of the left half, then of the right half: sensitive to mirroring."""

    spec = "fake:lr"
    cache_version = "test"

    @property
    def dim(self):
        return 6

    def _embed_image_batch(self, images):
        rows = []
        for im in images:
            a = np.asarray(im, np.float32)
            w = a.shape[1] // 2
            rows.append(np.concatenate([a[:, :w].reshape(-1, 3).mean(0), a[:, w:].reshape(-1, 3).mean(0)]) + 1)
        v = np.stack(rows)
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


class TextColor(ColorEmbedder):
    """Colour embedder that also embeds texts naming a colour (red, green or blue)."""

    supports_text = True

    def _embed_text_batch(self, texts):
        words = {"red": RED, "green": GREEN, "blue": BLUE}
        v = np.array([[float(c) + 1.0 for c in next(rgb for w, rgb in words.items() if w in t)] for t in texts])
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture(name="text_spec")
def fixture_text_spec():
    """A text-capable colour embedder registered under a CLIP-like spec."""
    register_embedder("openclip:fake-text", TextColor())
    yield "openclip:fake-text"
    unregister_embedder("openclip:fake-text")


def halves(left, right, size=8):
    """PNG bytes with the left half one colour and the right half another."""
    a = np.zeros((size, size, 3), np.uint8)
    a[:, : size // 2] = left
    a[:, size // 2 :] = right
    import io  # pylint: disable=import-outside-toplevel

    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


# ---- flip averaging -----------------------------------------------------
def test_flip_averages_with_the_mirror_image(ws):
    """With flip, an image and its mirror get the same vector, cached under a separate key."""
    register_embedder("fake:lr", LeftRight())
    try:
        pipe = Pipeline(ws)
        a, b = halves(RED, BLUE), halves(BLUE, RED)
        plain = pipe.embed_images("fake:lr", [("a", a), ("b", b)])
        assert not np.allclose(plain[0], plain[1])
        flipped = pipe.embed_images("fake:lr", [("a", a), ("b", b)], flip=True)
        assert np.allclose(flipped[0], flipped[1])
        assert np.allclose(np.linalg.norm(flipped, axis=1), 1)
        assert set(pipe.cache.get_many(embedders.cache_namespace("fake:lr"), ["a", "a" + FLIP_KEY])) == {
            "a", "a" + FLIP_KEY}
        assert pipe.embed_images("fake:lr", []).shape == (0, 0)

        t = ws.create("t", embedder="fake:lr")
        t.add_example(halves(RED, BLUE), "rb")
        t.add_example(halves(GREEN, GREEN), "g")
        t.update_settings(flip=True)
        p = pipe.predict(t, [halves(BLUE, RED)])[0]
        assert p.label == "rb"
        assert p.similarity == pytest.approx(1.0, abs=1e-5)  # the mirror matches exactly
    finally:
        unregister_embedder("fake:lr")


# ---- prompts, multi-label, thresholds -----------------------------------
def test_prompt_ensembles_are_averaged(ws, text_spec):
    """A description with several lines is embedded as the normalised mean of its prompts."""
    pipe = Pipeline(ws)
    v = pipe.embed_descriptions(text_spec, ["a red thing\na blue thing", "red"])
    red, blue = pipe.embed_texts(text_spec, ["red", "blue"])
    want = (red + blue) / np.linalg.norm(red + blue)
    assert np.allclose(v[0], want, atol=1e-6)
    assert np.allclose(v[1], red)
    assert pipe.embed_descriptions(text_spec, []).shape == (0, 0)


def test_multi_label_and_top_k(ws):
    """With a threshold, every class whose own similarity reaches it is listed (best first);
    top_k trims the scores, which come highest first."""
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    t.add_example(png((220, 120, 30)), "orange")
    t.add_example(png(BLUE), "blue")
    pipe = Pipeline(ws)
    q = png((220, 80, 30))  # between red and orange
    p = pipe.predict(t, [q])[0]
    assert p.labels == [p.label]  # no threshold: just the label
    assert list(p.scores) == sorted(p.scores, key=lambda c: -p.scores[c])
    t.update_settings(reject_below=0.97)
    p = pipe.predict(t, [q], top_k=2)[0]
    assert sorted(p.labels) == ["orange", "red"]
    assert p.labels[0] == p.label
    assert len(p.scores) == 2
    far = pipe.predict(t, [png((128, 128, 128))])[0]
    assert far.rejected
    assert far.labels == []


def test_zero_shot_threshold(ws, text_spec):
    """Zero-shot predictions use reject_below_zero_shot, not reject_below."""
    t = ws.create("t", embedder=text_spec)
    t.upsert_class("red", "a red thing")
    t.upsert_class("blue", "a blue thing")
    t.update_settings(reject_below=0.9999)
    pipe = Pipeline(ws)
    q = png((200, 90, 60))
    assert pipe.predict(t, [q])[0].label == "red"  # the image threshold doesn't apply
    t.update_settings(reject_below_zero_shot=0.9999)
    assert pipe.predict(t, [q])[0].rejected


def test_one_class_task(ws):
    """A single class predicts that class; its threshold separates members from outsiders."""
    t = ws.create("t", embedder="fake:color")
    for i in range(3):
        t.add_example(png(RED, noise=10, seed=i), "red")
    pipe = Pipeline(ws)
    t.update_settings(reject_below=0.99)
    member, outsider = pipe.predict(t, [png(RED, seed=9), png(BLUE)])
    assert member.label == "red"
    assert outsider.rejected
    r = evaluate_task(pipe, t)
    assert r["accuracy"] == 1.0
    assert r["suggested_reject_below"] > 0.99
    assert r["top_k_accuracy"] == {}


def test_predict_reports_progress(ws):
    """predict calls progress with the images embedded."""
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    seen = []
    Pipeline(ws).predict(t, [png(BLUE), png(GREEN)], progress=lambda d, n: seen.append((d, n)))
    assert seen == [(2, 2)]


def test_near_duplicates(ws):
    """Near-identical examples are paired, most similar first, marking different labels."""
    t = ws.create("t", embedder="fake:color")
    a = t.add_example(png(RED), "red")
    b = t.add_example(png(RED, noise=2, seed=1), "red")
    c = t.add_example(png(RED, noise=2, seed=2), "blue")
    t.add_example(png(GREEN), "green")
    pairs = Pipeline(ws).near_duplicates(t, threshold=0.999)
    assert {frozenset((d["a"], d["b"])) for d in pairs} == {frozenset(x) for x in ((a.id, b.id), (a.id, c.id),
                                                                                   (b.id, c.id))}
    assert [d["similarity"] for d in pairs] == sorted((d["similarity"] for d in pairs), reverse=True)
    mixed = next(d for d in pairs if {d["a"], d["b"]} == {a.id, c.id})
    assert not mixed["same_label"]
    assert set(mixed["labels"]) == {"red", "blue"}
    assert len(Pipeline(ws).near_duplicates(t, threshold=0.999, limit=1)) == 1


# ---- evaluation ---------------------------------------------------------
def _slow_leave_one_out(data, cfg):
    out = ev.HeldOut.empty(len(data.y), len(data.classes))
    for test in ev._folds(data.y, len(data.y), 0):  # pylint: disable=protected-access
        ev._fold(data, cfg, test, out)  # pylint: disable=protected-access
    return out


@pytest.mark.parametrize(("classifier", "params"), [("prototype", {}), ("prototype", {"scale": 5.0}),
                                                    ("knn", {"k": 3}), ("knn", {"k": 50})])
@pytest.mark.parametrize("labels", [[0, 1, 2, 0, 1, 2, 0, 1, 0], [0, 0, 0, 1, 1, 2, 2, 2, 3], [0] * 5, [0, 1]])
def test_fast_leave_one_out_matches_refitting(classifier, params, labels):
    """The closed-form leave-one-out gives the same results as refitting per example."""
    rng = np.random.default_rng(len(labels))
    X = rng.normal(size=(len(labels), 6)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.array(labels)
    data = ClassData([f"c{i}" for i in range(y.max() + 1)], X, y)
    cfg = TaskConfig(name="t", classifier=classifier, classifier_params=params)
    fast = ev._fast_leave_one_out(data, cfg, None)  # pylint: disable=protected-access
    slow = _slow_leave_one_out(data, cfg)
    assert np.array_equal(fast.pred, slow.pred)
    assert np.array_equal(fast.rank, slow.rank)
    assert np.allclose(fast.conf, slow.conf, atol=1e-5)
    scored = slow.pred >= 0
    assert np.allclose(fast.sim[scored], slow.sim[scored], atol=1e-5)


@pytest.mark.parametrize(("k", "at", "labels"), [(2, [1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0]),
                                                 (1, [0, 2, 2, 0, 0, 0, 0], [1, 1, 1, 0, 2, 2, 1]),
                                                 (3, [2, 2, 2, 0, 2, 2, 0, 0], [0, 2, 2, 1, 1, 2, 0, 2])])
def test_fast_knn_leave_one_out_matches_refitting_on_ties(k, at, labels):
    """With exact ties (repeated vectors), the closed-form kNN leave-one-out still matches refitting:
    both give tied neighbours to the lower column, and taking a column out keeps that order."""
    data = ClassData(["a", "b", "c"][: max(labels) + 1], np.eye(3, dtype=np.float32)[at], np.array(labels))
    cfg = TaskConfig(name="t", classifier="knn", classifier_params={"k": k})
    fast = ev._fast_leave_one_out(data, cfg, None)  # pylint: disable=protected-access
    slow = _slow_leave_one_out(data, cfg)
    assert np.array_equal(fast.pred, slow.pred)
    assert np.allclose(fast.conf, slow.conf, atol=1e-5)


def test_top_k_breaks_ties_by_column():
    """``top_k`` takes the largest entries, and the lowest columns among equal ones."""
    S = np.array([[1.0, 3, 3, 3, 0], [5, 4, 3, 2, 1], [0, 0, 0, 0, 0]])
    assert top_k(S, 2).tolist() == [[1, 2], [0, 1], [0, 1]]
    assert top_k(S, 5).tolist() == [[0, 1, 2, 3, 4]] * 3


def test_fast_knn_with_a_single_example(ws):
    """One example leaves nothing to vote with: nothing is scored."""
    t = ws.create("t", embedder="fake:color", classifier="knn")
    t.add_example(png(RED), "red")
    assert evaluate_task(Pipeline(ws), t)["n_scored"] == 0


def test_report_metrics(ws):
    """The report has macro-F1, top-k accuracy, every held-out prediction and a threshold suggestion."""
    t = ws.create("t", embedder="fake:color", classifier="logreg")
    for label, rgb in (("red", RED), ("green", GREEN), ("blue", BLUE), ("yellow", (220, 220, 30))):
        for i in range(3):
            t.add_example(png(rgb, noise=20, seed=i), label)
    t.add_example(png(GREEN, noise=20, seed=7), "red")  # a mislabelled one
    r = evaluate_task(Pipeline(ws), t)
    assert r["method"] == "leave-one-out"
    assert 0 < r["macro_f1"] < 1
    assert set(r["top_k_accuracy"]) == {"2", "3"}  # 5 is not below 4 classes
    assert r["top_k_accuracy"]["3"] >= r["top_k_accuracy"]["2"] >= r["accuracy"]
    assert len(r["predictions"]) == r["n_scored"] == 13
    assert {p["id"] for p in r["predictions"] if not p["correct"]} == {m["id"] for m in r["mistakes"]}
    assert -1 <= r["suggested_reject_below"] <= 1


def test_folds_that_leave_out_a_class(ws):
    """A fold whose training part lacks enough classes leaves its examples unscored; with nothing
    correct there is no threshold suggestion and no macro-F1."""
    t = ws.create("t", embedder="fake:color")
    for i in range(3):
        t.add_example(png(RED, noise=10, seed=i), "red")
    t.add_example(png(BLUE), "blue")
    r = evaluate_task(Pipeline(ws), t, folds=2)
    assert r["n_scored"] < 4
    assert ev._macro_f1(np.zeros((2, 2), int)) is None  # pylint: disable=protected-access
    empty = ev.HeldOut.empty(2, 2)
    assert ev._suggest_threshold(empty, np.array([0, 1])) is None  # pylint: disable=protected-access


def test_benchmark_flip_and_prompt_ensembles(ws, tmp_path, text_spec):
    """bench passes flip through and averages several prompts per class."""
    for c, rgb in (("red", RED), ("blue", BLUE)):
        (tmp_path / "d" / c).mkdir(parents=True)
        for i in range(3):
            (tmp_path / "d" / c / f"{i}.png").write_bytes(png(rgb, noise=10, seed=i))
    r = ev.benchmark(Pipeline(ws), tmp_path / "d", text_spec, way=2, shot=0, query=2, episodes=3,
                     text_template="a {}\nthe {} one", flip=True)
    assert r["accuracy"] == 1.0
    assert r["flip"] is True


# ---- devices and out-of-memory -----------------------------------------
def test_device_from_environment_and_mps(monkeypatch):
    """FEWSHOT_DEVICE wins (and must be a torch device); otherwise CUDA, then MPS, then CPU."""
    monkeypatch.setenv("FEWSHOT_DEVICE", "cpu")
    assert embedders._device() == "cpu"  # pylint: disable=protected-access
    monkeypatch.setenv("FEWSHOT_DEVICE", "warp-drive")
    with pytest.raises(ValueError, match="not a torch device"):
        embedders._device()  # pylint: disable=protected-access
    monkeypatch.delenv("FEWSHOT_DEVICE")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert embedders._device() == "mps"  # pylint: disable=protected-access
    with embedders._autocast("mps"):  # pylint: disable=protected-access
        assert not torch.is_autocast_enabled("cpu")


class Flaky(ColorEmbedder):
    """Runs out of memory on batches larger than ``fits``, raising ``error``."""

    def __init__(self, fits, error):
        super().__init__()
        self.fits, self.error, self.batches = fits, error, []

    def _embed_prepared(self, batch):
        if len(batch) > self.fits:
            raise self.error
        self.batches.append(len(batch))
        return super()._embed_prepared(batch)


@pytest.mark.parametrize("error", [torch.cuda.OutOfMemoryError("CUDA out of memory"),
                                   RuntimeError("MPS backend out of memory")])
def test_out_of_memory_halves_the_batch(error, monkeypatch):
    """On OOM the batch is split in halves and later batches stay smaller; results are unchanged."""
    freed = []
    monkeypatch.setattr(embedders, "_free_gpu_memory", lambda: freed.append(1))
    emb = Flaky(3, error)
    emb.batch_size = 8
    images = [Image.new("RGB", (4, 4), (i * 20, 50, 100)) for i in range(10)]
    got = emb.embed_images(images)
    assert np.allclose(got, ColorEmbedder().embed_images(images))
    assert emb.batch_size <= 3
    assert max(emb.batches) <= 3
    assert freed


def test_other_errors_and_single_image_oom_are_raised(monkeypatch):
    """Errors other than OOM, and OOM on a single image, propagate."""
    monkeypatch.setattr(embedders, "_free_gpu_memory", lambda: None)
    with pytest.raises(RuntimeError, match="boom"):
        Flaky(0, RuntimeError("boom")).embed_images([Image.new("RGB", (4, 4))] * 2)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        Flaky(0, torch.cuda.OutOfMemoryError("out of memory")).embed_images([Image.new("RGB", (4, 4))])


# ---- fetching URLs ------------------------------------------------------
class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves /img, /redirect and 404s."""

    def do_GET(self):  # pylint: disable=invalid-name  # http.server's naming
        """Answer one GET."""
        if self.path == "/img":
            body = png(RED)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/img")
            self.end_headers()
        else:
            self.send_error(404, "Not Found")

    def log_message(self, *_args):
        pass


@pytest.fixture(name="server")
def fixture_server():
    """A local HTTP server: /img is a PNG, /redirect goes to /img, anything else is 404."""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_fetch_url(server):
    """Bodies are fetched up to a size limit; HTTP and network errors become ValueErrors."""
    assert fetch.fetch_url(server + "/img", 10_000) == png(RED)
    assert fetch.fetch_url(server + "/redirect", 10_000) == png(RED)
    with pytest.raises(ValueError, match="larger than"):
        fetch.fetch_url(server + "/img", 10)
    with pytest.raises(ValueError, match="HTTP 404"):
        fetch.fetch_url(server + "/nope", 10_000)
    with pytest.raises(ValueError, match="Not an http"):
        fetch.fetch_url("file:///etc/passwd", 10_000)
    with pytest.raises(ValueError, match="Could not fetch"):
        fetch.fetch_url("http://127.0.0.1:1/x", 10_000)


def test_fetch_public_only(server, monkeypatch):
    """public_only refuses private hosts, and redirects are checked too."""
    with pytest.raises(ValueError, match="not a public address"):
        fetch.fetch_url(server + "/img", 10_000, public_only=True)
    real_check = fetch.check_public
    monkeypatch.setattr(fetch, "check_public", lambda url: None if url.endswith("/redirect") else real_check(url))
    with pytest.raises(ValueError, match="not a public address"):
        fetch.fetch_url(server + "/redirect", 10_000, public_only=True)


def test_check_public(monkeypatch):
    """Only http(s) URLs whose host resolves to public addresses pass."""
    real = socket.getaddrinfo

    def no_dns(host, *args, **kw):  # IP literals resolve without DNS; names fail at once
        if host[0].isdigit() or ":" in host:
            return real(host, *args, **kw)
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    fetch.check_public("http://8.8.8.8/x.png")
    for url, msg in (("ftp://8.8.8.8/x", "Not an http"), ("http:///x", "Not an http"),
                     ("http://10.0.0.1/x", "not a public"), ("http://[::1]/x", "not a public"),
                     ("http://no-such-host.invalid/x", "Can't resolve")):
        with pytest.raises(ValueError, match=msg):
            fetch.check_public(url)
    assert fetch.is_url("HTTPS://x")
    assert not fetch.is_url("/tmp/x.png")


def test_fetch_connects_to_the_checked_address(server, monkeypatch):
    """Each connection goes to the address that passed the check, looked up once: the host name
    itself is never resolved again (here it can't be resolved at all), and redirects are pinned too."""
    port = int(server.rsplit(":", 1)[1])
    looked_up = []

    def checked(host, _port):
        looked_up.append(host)
        return ["127.0.0.1"]  # pretend this passed the public-address check

    monkeypatch.setattr(fetch, "public_addresses", checked)
    url = f"http://pinned.invalid:{port}"
    assert fetch.fetch_url(url + "/redirect", 10_000, public_only=True) == png(RED)
    assert looked_up == ["pinned.invalid"] * 3  # the check, the first connection, the redirect's


def test_fetch_refuses_dns_rebinding(server, monkeypatch):
    """A host that answers the check with a public address and then with a private one is refused."""
    port = int(server.rsplit(":", 1)[1])
    answers = iter(["93.184.216.34", "127.0.0.1"])

    def rebinding(_host, p, *_args, **_kw):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), p))]

    monkeypatch.setattr(socket, "getaddrinfo", rebinding)
    with pytest.raises(ValueError, match="not a public address"):
        fetch.fetch_url(f"http://rebind.test:{port}/img", 10_000, public_only=True)


def test_write_atomic_cleans_up(tmp_path, monkeypatch):
    """A failed atomic write leaves neither the file nor a temporary file."""
    import os  # pylint: disable=import-outside-toplevel

    from fewshot.fsutil import write_atomic  # pylint: disable=import-outside-toplevel

    def broken(*_):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        write_atomic(tmp_path / "x", b"data")
    assert not list(tmp_path.iterdir())


def test_fetch_https_is_pinned_too(server, monkeypatch):
    """HTTPS connections also go to the checked address (here a plain-HTTP server, so the TLS
    handshake fails after connecting to it)."""
    port = int(server.rsplit(":", 1)[1])
    looked_up: list[str] = []

    def checked(host, _port):
        looked_up.append(host)
        return ["127.0.0.1"]

    monkeypatch.setattr(fetch, "public_addresses", checked)
    with pytest.raises(ValueError, match="Could not fetch"):
        fetch.fetch_url(f"https://pinned.invalid:{port}/img", 10_000, public_only=True)
    assert looked_up == ["pinned.invalid", "pinned.invalid"]


# ---- second review: wrong results ---------------------------------------
def test_ties_count_against_the_true_class():
    """A true class tied with others is ranked below all of them, so ties never earn top-k credit."""
    probs = np.array([[0.5, 0.25, 0.25, 0.0], [0.4, 0.2, 0.2, 0.2]])
    assert ev._ranks(probs, np.array([1, 3]), np.array([True, True])).tolist() == [2, 3]  # pylint: disable=protected-access


def test_top_k_is_not_inflated_by_knn_ties(ws):
    """Random labels with kNN (k=1) give chance-level top-2, not 100%."""
    t = ws.create("t", embedder="fake:color", classifier="knn", classifier_params={"k": 1})
    rng = np.random.default_rng(0)
    for i in range(40):
        t.add_example(png(tuple(int(c) for c in rng.integers(0, 255, 3)), seed=i), f"c{rng.integers(5)}")
    r = evaluate_task(Pipeline(ws), t)
    assert r["top_k_accuracy"]["2"] < 0.9


def test_threshold_applies_to_the_predicted_class():
    """When a description tips the vote to a class the image is less similar to, predict reports
    that class's own similarity, so a threshold between the two rejects it instead of accepting
    it below the line."""
    from fewshot.classifiers import make_classifier  # pylint: disable=import-outside-toplevel
    from fewshot.pipeline import _to_pred  # pylint: disable=import-outside-toplevel

    X = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    data = ClassData(["a", "b"], X, np.array([0, 1]), {1: np.array([0.6, 0.8, 0], np.float32)})
    clf = make_classifier("prototype", text_weight=3.0).fit(data)
    q = np.array([[0.8, 0.6, 0]], np.float32)
    probs, sims = clf.predict(q)
    per_class = clf.class_similarity(q)[0]
    assert per_class[0] > per_class[1]  # nearer to a...
    assert probs[0].argmax() == 1  # ...but b's description wins
    assert sims[0] == pytest.approx(per_class[1])
    p = _to_pred(clf, np.array([0.7, 0.7]), (probs[0], float(sims[0]), per_class), None)
    assert (p.rejected, p.label, p.labels) == (True, None, ["a"])  # rejected as b, but a clears its threshold


def test_top_k_must_be_positive(ws):
    """top_k below 1 is a ValueError, not a silent trim from the end."""
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        Pipeline(ws).predict(t, [png(RED)], top_k=0)


def test_near_duplicates_stays_bounded_and_finds_identical(ws):
    """Only the best ``limit`` pairs are kept while scanning, and threshold 1.0 still finds
    embeddings that are identical up to float rounding."""
    t = ws.create("t", embedder="fake:color")
    for size in (8, 9, 10):  # same colour, different files: identical embeddings
        t.add_example(png(RED, size=size), "red")
    t.add_example(png(GREEN), "green")
    pipe = Pipeline(ws)
    assert len(pipe.near_duplicates(t, threshold=1.0)) == 3
    assert len(pipe.near_duplicates(t, threshold=0.5, limit=2)) == 2


class _Activation:  # pylint: disable=too-few-public-methods
    """Stands in for a forward pass's GPU tensors."""


class Leaky(ColorEmbedder):
    """Runs out of memory on batches over 2; records weak references to each pass's 'activations'."""

    def __init__(self):
        super().__init__()
        self.refs = []
        self.alive_at_retry = []

    def _embed_prepared(self, batch):
        import gc  # pylint: disable=import-outside-toplevel
        import weakref  # pylint: disable=import-outside-toplevel

        gc.collect()
        self.alive_at_retry.append(any(r() is not None for r in self.refs))
        act = _Activation()  # held by this frame, so by a traceback through it
        self.refs.append(weakref.ref(act))
        if len(batch) > 2:
            raise RuntimeError("CUDA out of memory")
        return super()._embed_prepared(batch)


def test_out_of_memory_retry_frees_the_failed_pass(monkeypatch):
    """The halves run after the failed pass's memory is released, not while its traceback holds it."""
    monkeypatch.setattr(embedders, "_free_gpu_memory", lambda: None)
    emb = Leaky()
    emb.batch_size = 8
    emb.embed_images([Image.new("RGB", (4, 4), (i * 30, 0, 0)) for i in range(8)])
    assert not any(emb.alive_at_retry)


def test_texts_are_embedded_in_batches_with_oom_halving(monkeypatch):
    """Texts go through the model batch_size at a time, and halve on OOM like images."""
    monkeypatch.setattr(embedders, "_free_gpu_memory", lambda: None)
    sizes = []

    class Texts(TextColor):
        """Text embedder that runs out of memory above 2 texts."""

        def _embed_text_batch(self, texts):
            if len(texts) > 2:
                raise RuntimeError("CUDA out of memory")
            sizes.append(len(texts))
            return super()._embed_text_batch(texts)

    emb = Texts()
    emb.batch_size = 4
    v = emb.embed_texts(["red", "green", "blue", "red", "blue"])
    assert v.shape == (5, 3)
    assert sizes == [2, 2, 1]
    assert emb.batch_size == 2


def test_nat64_and_mapped_private_addresses_are_refused(monkeypatch):
    """An IPv6 address that reaches a private IPv4 one through NAT64 is not public."""
    def answer(ip):
        return lambda host, port, *a, **k: [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, port, 0, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", answer("64:ff9b::7f00:1"))
    with pytest.raises(ValueError, match="not a public address"):
        fetch.public_addresses("nat64.test", 80)
    monkeypatch.setattr(socket, "getaddrinfo", answer("64:ff9b::808:808"))  # 8.8.8.8
    assert fetch.public_addresses("nat64.test", 80) == ["64:ff9b::808:808"]

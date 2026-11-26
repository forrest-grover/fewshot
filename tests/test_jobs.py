"""Background jobs: evaluation jobs over the API, cancelling, limits, sharing and forgetting."""

import threading
import time
import uuid

import pytest

import fewshot.api
import fewshot.evaluate
from fewshot.jobs import Job, Jobs, TooManyJobs
from tests.conftest import api_task, wait_for_job


# ---- over the API --------------------------------------------------------
def test_evaluation_job(client):
    """An evaluation job reports progress and ends with the same report as the direct endpoint."""
    task = api_task(client)
    r = client.post(f"/api/tasks/{task}/evaluate/jobs")
    assert r.status_code == 202
    job = wait_for_job(client, r.json()["id"])
    assert job["status"] == "done"
    assert (job["task"], job["kind"], job["stage"]) == (task, "evaluate", "evaluating")
    assert job["done"] == job["total"] == 6  # leave-one-out: one fold per example
    assert job["result"] == client.post(f"/api/tasks/{task}/evaluate").json()
    assert client.get("/api/health").json()["jobs_running"] == 0


def test_evaluation_job_errors(client):
    """Bad arguments fail at once; errors during the work are reported on the job."""
    assert client.post("/api/tasks/nope/evaluate/jobs").status_code == 404
    client.post("/api/tasks", json={"name": "empty", "embedder": "fake:color"})
    assert client.post("/api/tasks/empty/evaluate/jobs", params={"folds": 1}).status_code == 400
    job = wait_for_job(client, client.post("/api/tasks/empty/evaluate/jobs").json()["id"])
    assert (job["status"], job["error"]) == ("failed", "Task has no examples to evaluate")
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.delete("/api/jobs/nope").status_code == 404


def test_job_internal_error_and_cancel(client, monkeypatch):
    """An unexpected exception fails the job; cancelling stops it at its next progress step."""
    task = api_task(client, n=1)

    def broken(*_args, **_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(fewshot.evaluate, "evaluate_task", broken)
    job = wait_for_job(client, client.post(f"/api/tasks/{task}/evaluate/jobs").json()["id"])
    assert (job["status"], job["error"]) == ("failed", "Internal error; see the server log")

    started = threading.Event()

    def slow(_pipe, _task, _folds, progress):
        for i in range(1000):
            progress("evaluating", i, 1000)
            started.set()
            time.sleep(0.01)

    monkeypatch.setattr(fewshot.evaluate, "evaluate_task", slow)
    jid = client.post(f"/api/tasks/{task}/evaluate/jobs").json()["id"]
    assert started.wait(5)
    assert client.get("/api/health").json()["jobs_running"] == 1
    assert client.delete(f"/api/jobs/{jid}").status_code == 200
    job = wait_for_job(client, jid)
    assert job["status"] == "cancelled"
    assert job["done"] < 999  # stopped long before the end


def test_old_jobs_are_forgotten(client, monkeypatch):
    """Only the most recent `kept` finished jobs are remembered."""
    task = api_task(client, n=1)
    monkeypatch.setattr(client.srv.jobs, "kept", 2)
    ids = []
    for _ in range(3):
        ids.append(client.post(f"/api/tasks/{task}/evaluate/jobs").json()["id"])
        wait_for_job(client, ids[-1])
    assert client.get(f"/api/jobs/{ids[0]}").status_code == 404
    assert all(client.get(f"/api/jobs/{j}").status_code == 200 for j in ids[1:])


def test_jobs_are_capped_and_shared(client, monkeypatch):
    """The same evaluation already running is shared; past MAX_JOBS running jobs, 429."""
    task = api_task(client, n=1)
    release = threading.Event()

    def slow(_pipe, _task, _folds, progress):
        progress("evaluating", 0, 1)
        release.wait(5)

    monkeypatch.setattr(fewshot.evaluate, "evaluate_task", slow)
    monkeypatch.setattr(fewshot.api, "MAX_JOBS", 1)
    first = client.post(f"/api/tasks/{task}/evaluate/jobs").json()
    assert client.post(f"/api/tasks/{task}/evaluate/jobs").json()["id"] == first["id"]  # shared
    r = client.post(f"/api/tasks/{task}/evaluate/jobs", params={"folds": 2})
    assert r.status_code == 429
    release.set()
    wait_for_job(client, first["id"])


def test_a_recreated_task_doesnt_get_the_old_tasks_job(client, monkeypatch):
    """Jobs are shared by the task's uid, not its name: a task deleted and made again under the
    same name, at the same revision, gets its own evaluation."""
    task = api_task(client, n=1)
    release = threading.Event()
    seen = []
    real = fewshot.evaluate.evaluate_task

    def slow(pipe, t, folds, progress):
        seen.append(t.config.uid)
        release.wait(5)
        return real(pipe, t, folds, progress=progress)

    monkeypatch.setattr(fewshot.evaluate, "evaluate_task", slow)
    first = client.post(f"/api/tasks/{task}/evaluate/jobs").json()
    revision = client.get(f"/api/tasks/{task}").json()["revision"]
    assert client.delete(f"/api/tasks/{task}").status_code == 204
    assert api_task(client, n=1) == task
    assert client.get(f"/api/tasks/{task}").json()["revision"] == revision
    second = client.post(f"/api/tasks/{task}/evaluate/jobs").json()
    assert second["id"] != first["id"]
    release.set()
    assert [wait_for_job(client, j["id"])["status"] for j in (first, second)] == ["done", "done"]
    assert len(set(seen)) == 2


def test_a_cancelled_evaluation_is_not_shared(client, monkeypatch):
    """An evaluation asked to stop, though still running, isn't handed to the next request."""
    task = api_task(client, n=1)
    release = threading.Event()

    def slow(_pipe, _task, _folds, progress):
        release.wait(5)
        progress("evaluating", 0, 1)  # raises Cancelled

    monkeypatch.setattr(fewshot.evaluate, "evaluate_task", slow)
    first = client.post(f"/api/tasks/{task}/evaluate/jobs").json()["id"]
    assert client.delete(f"/api/jobs/{first}").status_code == 200
    second = client.post(f"/api/tasks/{task}/evaluate/jobs").json()["id"]
    assert second != first
    client.delete(f"/api/jobs/{second}")
    release.set()
    assert [wait_for_job(client, j)["status"] for j in (first, second)] == ["cancelled", "cancelled"]


@pytest.mark.usefixtures("task_t")
def test_unknown_jobs_say_why(client):
    """A job id the server doesn't know may be from before a restart; the message says so."""
    r = client.get("/api/jobs/abc")
    assert r.status_code == 404
    assert "server was restarted" in r.json()["detail"]


# ---- Jobs ----------------------------------------------------------------
def test_job_limit_is_checked_with_the_start():
    """Jobs.start refuses past ``limit``, even when many starts race."""
    jobs = Jobs()
    gate = threading.Event()
    started, refused = [], []

    def start():
        try:
            job = Job(task="t", kind="k", key=uuid.uuid4().hex)  # different work, not shared
            started.append(jobs.start(job, lambda _p: gate.wait(5), limit=3))
        except TooManyJobs:
            refused.append(1)

    threads = [threading.Thread(target=start) for _ in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert (len(started), len(refused)) == (3, 17)
    gate.set()
    deadline = time.time() + 5
    while jobs.running() and time.time() < deadline:
        time.sleep(0.01)
    assert jobs.start(Job(task="t", kind="k"), lambda _p: None, limit=3).status in {"running", "done"}


def test_the_same_job_is_shared_under_the_lock():
    """Jobs.start itself returns the running job doing the same work, so racing starts share it."""
    jobs, gate = Jobs(), threading.Event()
    got = []

    def start():
        got.append(jobs.start(Job(task="t", kind="evaluate", revision=3), lambda _p: gate.wait(5), limit=4).id)

    threads = [threading.Thread(target=start) for _ in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(set(got)) == 1
    other = jobs.start(Job(task="t", kind="evaluate", revision=4), lambda _p: gate.wait(5))
    assert other.id != got[0]  # another version is other work
    gate.set()


def test_a_cancelled_job_frees_its_place_at_once():
    """A job asked to stop is "cancelling" and no longer counts towards the limit."""
    jobs, gate = Jobs(), threading.Event()
    first = jobs.start(Job(task="t", kind="a"), lambda _p: gate.wait(5), limit=1)
    first.request_cancel()
    assert first.to_dict()["status"] == "cancelling"
    second = jobs.start(Job(task="t", kind="b"), lambda _p: gate.wait(5), limit=1)
    assert second.status == "running"
    assert jobs.running() == [second]
    gate.set()
    deadline = time.time() + 5
    while first.finished is None and time.time() < deadline:
        time.sleep(0.01)
    assert first.status == "done"  # it finished without another progress step: its result stands
    first.request_cancel()  # too late: nothing changes
    assert first.status == "done"


def test_only_recent_results_are_kept():
    """Finished jobs past kept_results keep their status but not their (possibly large) result."""
    jobs = Jobs(kept=5, kept_results=2)
    def work(i):
        return lambda _p: {"n": i}

    done = [jobs.start(Job(task="t", kind=str(i)), work(i)) for i in range(4)]
    deadline = time.time() + 5
    while any(j.finished is None for j in done) and time.time() < deadline:
        time.sleep(0.01)
    gate = threading.Event()
    jobs.start(Job(task="t", kind="last"), lambda _p: gate.wait(5))  # starting a job forgets old results too
    kept = [jobs.get(j.id) for j in done]
    gate.set()
    assert [j.expired for j in kept if j] == [True, True, False, False]
    assert kept[0] is not None
    assert kept[0].result is None
    assert kept[0].to_dict()["expired"] is True

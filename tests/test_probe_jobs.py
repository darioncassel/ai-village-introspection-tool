"""Unit tests for the probe log / stars Store and the Jobs queue: durability of the JSON-lines logs
(line separators, torn or corrupt lines, failed writes), job lifecycle ordering, and thread safety.

Synthetic prompts and fake model calls only (no dataset, no API)."""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from village_introspect import probe_jobs as pj
from village_introspect import village_lib as vl

SEPS = "\u2028\u2029\u0085"  # characters str.splitlines() treats as line breaks; json writes them raw


def fake_prompt(params: dict) -> dict:
    if "ei" not in params:
        raise ValueError("missing parameter: ei")
    system = "You are an agent."
    msgs = [{"t": "user", "text": params.get("context", "hello")}]
    return {
        "resolved": {"target": "tierb", "agent": "a1", "agent_id": "a1", "ei": params["ei"], "day": 5},
        "mode": "belief",
        "model": "fake-model",
        "fidelity": {"level": "approx"},
        "system": system,
        "messages": msgs,
        "sha256": vl.prompt_sha(system, msgs),
        "default_question": "why?",
    }


def answering(text="an answer", gate: threading.Event = None, calls: list = None):
    def call(system, messages, model):
        if gate is not None:
            assert gate.wait(10)
        if calls is not None:
            calls.append(messages)
        return {
            "thinking": "",
            "text": text,
            "tool_use": None,
            "usage": {"in": 1, "out": 1},
            "stop_reason": "end",
            "request": {"provider": "fake", "max_tokens": 16, "n": len(messages)},
        }

    return call


def failing(system, messages, model):
    raise RuntimeError("boom")


def run_all(jobs: pj.Jobs) -> None:
    """Wait for every submitted job, then give the Jobs a fresh pool for the next submit."""
    jobs.pool.shutdown(wait=True)
    jobs.pool = pj.ThreadPoolExecutor(max_workers=1)


def wait_status(jobs: pj.Jobs, job_id: str, want: str, timeout: float = 10) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = jobs.get(job_id)
        if v["status"] == want:
            return v
        time.sleep(0.005)
    raise AssertionError(f"job {job_id} never reached {want!r} (last: {v['status']!r})")


def warnings_for(err: str, path: Path) -> list:
    return [ln for ln in err.splitlines() if str(path) in ln and "skipped" in ln]


# ---- reload ----------------------------------------------------------------------------------------------
def test_records_with_line_separators_survive_a_restart(tmp_path):
    store = pj.Store(tmp_path)
    jobs = pj.Jobs(store, fake_prompt, answering(f"one{SEPS[0]}two{SEPS[1]}three{SEPS[2]}four"), workers=1)
    a = jobs.submit({"ei": 1, "question": f"pasted{SEPS}question", "context": f"a{SEPS}b"})
    run_all(jobs)
    f = jobs.submit({"parent_id": a["probe_id"], "question": f"and{SEPS[0]}then?"})
    run_all(jobs)
    star = store.star_add(probe_id=a["probe_id"], day=5, note=f"see{SEPS}here")
    before = {pid: store.get(pid) for pid in (a["probe_id"], f["probe_id"])}
    assert all(r["status"] == "done" for r in before.values())

    again = pj.Store(tmp_path)
    assert set(again.probes) == {a["probe_id"], f["probe_id"]}
    for pid, rec in before.items():
        assert again.get(pid) == rec
    assert again.get(f["probe_id"])["messages"][-2]["text"] == f"one{SEPS[0]}two{SEPS[1]}three{SEPS[2]}four"
    assert again.star_list() == [star] and again.star_list()[0]["note"] == f"see{SEPS}here"
    assert again.n_probes_by_ei() == {1: 1} and again.probes_by_day() == {5: 1}


def test_crlf_line_endings_load(tmp_path):
    rec = {"probe_id": "p1", "status": "done", "day": 2, "resolved": {"ei": 9}}
    (tmp_path / "probes.jsonl").write_bytes(json.dumps(rec).encode() + b"\r\n")
    (tmp_path / "stars.jsonl").write_bytes(json.dumps({"op": "add", "id": "s1", "ei": 9}).encode() + b"\r\n")
    store = pj.Store(tmp_path)
    assert store.get("p1") == rec and store.starred_eis() == {9}


def test_torn_last_line_is_skipped_and_does_not_swallow_the_next_record(tmp_path, capsys):
    good = json.dumps({"probe_id": "good1", "status": "done", "text": "café"}, ensure_ascii=False)
    torn = json.dumps({"probe_id": "torn", "status": "done", "text": "x" * 50}, ensure_ascii=False)[:30]
    (tmp_path / "probes.jsonl").write_text(good + "\n" + torn, encoding="utf-8")
    (tmp_path / "stars.jsonl").write_text(json.dumps({"op": "add", "id": "s1", "ei": 1}) + '\n{"op": "ad', "utf-8")

    store = pj.Store(tmp_path)
    err = capsys.readouterr().err
    assert set(store.probes) == {"good1"} and store.starred_eis() == {1}
    assert len(warnings_for(err, store.probes_path)) == 1 and len(warnings_for(err, store.stars_path)) == 1

    store.append_probe({"probe_id": "after", "status": "done"})
    store.star_add(ei=2)
    again = pj.Store(tmp_path)
    assert set(again.probes) == {"good1", "after"} and again.starred_eis() == {1, 2}
    assert again.get("good1")["text"] == "café"


def test_torn_line_left_by_a_failed_write_in_this_process_does_not_swallow_the_next_record(tmp_path):
    store = pj.Store(tmp_path)
    store.append_probe({"probe_id": "a", "status": "done"})
    with open(store.probes_path, "ab") as f:  # what an interrupted write leaves behind
        f.write(b'{"probe_id": "lost", "te')
    store.append_probe({"probe_id": "b", "status": "done"})
    assert set(pj.Store(tmp_path).probes) == {"a", "b"}


def test_complete_record_missing_only_its_newline_is_kept(tmp_path):
    (tmp_path / "probes.jsonl").write_text('{"probe_id": "a"}\n{"probe_id": "b"}', encoding="utf-8")
    store = pj.Store(tmp_path)
    store.append_probe({"probe_id": "c"})
    assert set(pj.Store(tmp_path).probes) == {"a", "b", "c"}


def test_invalid_utf8_blank_and_non_object_lines_never_stop_startup(tmp_path, capsys):
    ok = {"probe_id": "ok", "status": "done"}
    lines = [
        json.dumps(ok).encode(),
        b"",
        b"   ",
        b"123",
        b"null",
        b"[1, 2]",
        b'"a string"',
        b'{"no_id": true}',
        b'{"probe_id": ["not", "a", "string"]}',
        b'{"probe_id": "bad", "text": "\xff\xfe"}',  # invalid UTF-8 in a complete line
        b'{"probe_id": "cut", "text": "caf' + "é".encode()[:1],  # torn inside a multibyte character
    ]
    (tmp_path / "probes.jsonl").write_bytes(b"\n".join(lines))
    stars = [b'{"op": "add", "ei": 3}', b'{"op": "add", "id": ["x"], "ei": 4}', b'{"op": "remove", "id": {}}']
    stars += [b"7", b'{"op": "add", "id": "s1", "ei": 5}', b"\xc3"]
    (tmp_path / "stars.jsonl").write_bytes(b"\n".join(stars) + b"\n")

    store = pj.Store(tmp_path)
    err = capsys.readouterr().err
    assert set(store.probes) == {"ok"} and store.starred_eis() == {5}
    assert len(warnings_for(err, store.probes_path)) == 10  # one per unreadable line, blank ones included
    assert len(warnings_for(err, store.stars_path)) == 5


# ---- failed writes -----------------------------------------------------------------------------------------
def test_failed_probe_log_write_is_reported_on_the_job_and_logged(tmp_path, capsys):
    store = pj.Store(tmp_path)
    store.probes_path.mkdir()  # every append to the probe log now fails
    jobs = pj.Jobs(store, fake_prompt, answering("the answer"), workers=1)
    j = jobs.submit({"ei": 1})
    run_all(jobs)

    v = jobs.get(j["job_id"])
    assert v["status"] == "error"
    assert v["error"].startswith("answer received but couldn't be saved to the probe log: ")
    assert v["result"]["text"] == "the answer"  # the viewer can still show it
    assert v["result"]["save_error"].startswith("couldn't be saved to the probe log: ")  # the result explains itself
    assert store.get(j["probe_id"]) is None and store.version == 0
    assert "Traceback" in capsys.readouterr().err
    with pytest.raises(ValueError, match="wasn't saved"):
        jobs.submit({"parent_id": j["probe_id"], "question": "more?"})

    jobs.call_fn = failing  # a failed call whose record can't be saved keeps both errors
    e = jobs.submit({"ei": 1})
    run_all(jobs)
    ve = jobs.get(e["job_id"])
    assert ve["status"] == "error" and "boom" in ve["error"] and "couldn't be saved to the probe log" in ve["error"]

    store.probes_path.rmdir()  # writable again: the next answer is saved normally
    jobs.call_fn = answering("saved")
    k = jobs.submit({"ei": 1})
    run_all(jobs)
    assert jobs.get(k["job_id"])["status"] == "done" and "error" not in jobs.get(k["job_id"])
    assert set(pj.Store(tmp_path).probes) == {k["probe_id"]}


def test_failed_star_write_leaves_no_phantom_star(tmp_path):
    store = pj.Store(tmp_path)
    store.stars_path.mkdir()
    with pytest.raises(OSError):
        store.star_add(ei=3, day=5)
    assert store.starred_eis() == set() and store.star_list() == [] and store.version == 0
    store.stars_path.rmdir()
    s = store.star_add(ei=3, day=5)
    assert pj.Store(tmp_path).star_list() == [s]

    store.stars_path.rename(tmp_path / "stars.old")
    store.stars_path.mkdir()
    with pytest.raises(OSError):
        store.star_remove(ei=3)
    assert store.starred_eis() == {3}  # still starred, as it still is on disk


# ---- job lifecycle -----------------------------------------------------------------------------------------
def test_probe_record_carries_stop_reason_and_request(tmp_path):
    store = pj.Store(tmp_path)
    jobs = pj.Jobs(store, fake_prompt, answering(), workers=1)
    j = jobs.submit({"ei": 1})
    run_all(jobs)
    f = jobs.submit({"parent_id": j["probe_id"], "question": "more?"})
    run_all(jobs)
    rec, fr = store.get(j["probe_id"]), store.get(f["probe_id"])
    assert rec["stop_reason"] == "end" and rec["request"] == {"provider": "fake", "max_tokens": 16, "n": 1}
    assert fr["stop_reason"] == "end" and fr["request"]["n"] == 3  # its own call's, not the parent's
    assert pj.Store(tmp_path).get(j["probe_id"])["request"] == rec["request"]
    jobs.call_fn = failing
    e = jobs.submit({"ei": 1})
    run_all(jobs)
    er = store.get(e["probe_id"])
    assert er["status"] == "error" and er["stop_reason"] is None and er["request"] is None


def test_job_is_done_only_once_its_record_is_in_the_store(tmp_path):
    class SlowStore(pj.Store):
        def __init__(self, d):
            super().__init__(d)
            self.entered, self.gate = threading.Event(), threading.Event()

        def append_probe(self, rec):
            self.entered.set()
            assert self.gate.wait(10)
            super().append_probe(rec)

    store = SlowStore(tmp_path)
    jobs = pj.Jobs(store, fake_prompt, answering(), workers=1)
    j = jobs.submit({"ei": 1})
    assert store.entered.wait(10)
    v = jobs.get(j["job_id"])
    assert v["status"] == "running" and "result" not in v
    with pytest.raises(ValueError, match="wait for the previous answer to finish"):
        jobs.submit({"parent_id": j["probe_id"], "question": "more?"})
    store.gate.set()
    wait_status(jobs, j["job_id"], "done")
    assert store.get(j["probe_id"]) is not None
    jobs.submit({"parent_id": j["probe_id"], "question": "more?"})  # accepted the moment 'done' shows
    run_all(jobs)


def test_follow_up_while_the_previous_answer_is_still_running(tmp_path):
    gate = threading.Event()
    store = pj.Store(tmp_path)
    jobs = pj.Jobs(store, fake_prompt, answering(gate=gate), workers=1)
    a = jobs.submit({"ei": 1})
    b = jobs.submit({"ei": 2})
    wait_status(jobs, a["job_id"], "running")
    assert jobs.get(b["job_id"])["status"] == "queued"
    for parent in (a, b):
        with pytest.raises(ValueError, match="wait for the previous answer to finish"):
            jobs.submit({"parent_id": parent["probe_id"], "question": "and?"})
    with pytest.raises(ValueError, match="unknown parent_id"):
        jobs.submit({"parent_id": "nope", "question": "and?"})
    gate.set()
    run_all(jobs)
    f = jobs.submit({"parent_id": a["probe_id"], "question": "and?"})
    run_all(jobs)
    assert store.get(f["probe_id"])["parent_id"] == a["probe_id"]


def test_unknown_job_is_none(tmp_path):
    jobs = pj.Jobs(pj.Store(tmp_path), fake_prompt, answering(), workers=1)
    assert jobs.get("nope") is None
    j = jobs.submit({"ei": 1})
    run_all(jobs)
    assert jobs.get(j["job_id"])["status"] == "done" and jobs.get("nope") is None


# ---- thread safety -----------------------------------------------------------------------------------------
@pytest.fixture
def fast_switching():
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads as often as possible to expose unlocked iteration
    yield
    sys.setswitchinterval(old)


def _hammer(workers, readers, seconds=1.5) -> list:
    stop, errors = threading.Event(), []

    def loop(fn):
        while not stop.is_set():
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - any exception is a failure here
                errors.append(f"{getattr(fn, '__name__', fn)}: {type(e).__name__}: {e}")
                return

    threads = [threading.Thread(target=loop, args=(fn,), daemon=True) for fn in workers + readers]
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join(10)
    return errors


def test_store_readers_survive_concurrent_writes(tmp_path, fast_switching):
    store = pj.Store(tmp_path)
    n = iter(range(10**9))

    def write():
        i = next(n)
        store.append_probe({"probe_id": f"p{i}", "status": "done", "day": i % 7, "resolved": {"ei": i % 50}})
        store.star_add(ei=i, day=i % 7)
        if i % 3 == 0:
            store.star_remove(ei=i)

    readers = [
        store.n_probes_by_ei,
        store.probes_by_day,
        lambda: store.list(limit=5),
        lambda: store.list(day=3, limit=10**6),
        store.star_list,
        store.stars_by_day,
        store.starred_eis,
        lambda: store.get("p1"),
    ]
    errors = _hammer([write], readers)
    assert not errors, errors[:5]
    assert len(pj.Store(tmp_path).probes) == len(store.probes) > 0


def test_jobs_get_and_list_survive_concurrent_submits(tmp_path, fast_switching):
    gate = threading.Event()
    jobs = pj.Jobs(pj.Store(tmp_path), fake_prompt, answering(gate=gate), workers=1)
    first = jobs.submit({"ei": 0})
    queued = jobs.submit({"ei": 1})
    try:
        errors = _hammer(
            [lambda: jobs.submit({"ei": 2})],
            [lambda: jobs.get(first["job_id"]), lambda: jobs.get(queued["job_id"]), jobs.list],
            seconds=1.0,
        )
    finally:
        jobs.pool.shutdown(wait=False, cancel_futures=True)
        gate.set()
    assert not errors, errors[:5]

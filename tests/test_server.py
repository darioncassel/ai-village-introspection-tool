"""HTTP-level tests for server.py: a real Server on an ephemeral port, with the village index faked as ready
(no dataset, no keys, no model calls)."""

import http
import http.client
import inspect
import json
import os
import socketserver
import sys
import threading
import time
import types

import pytest

from village_introspect import probe_jobs as pj
from village_introspect import server
from village_introspect import village_lib as vl


def _pp(body):
    """A resolved probe prompt, standing in for village_lib.probe_prompt."""
    return {
        "resolved": {"target": "tierb", "agent": "a", "agent_id": "a"},
        "mode": "belief",
        "model": "m",
        "fidelity": {},
        "system": "s",
        "messages": [],
        "sha256": "0",
    }


@pytest.fixture()
def srv(tmp_path, monkeypatch):
    store = pj.Store(tmp_path)
    jobs = pj.Jobs(store, _pp, lambda system, messages, model: {"text": "answer"}, workers=1)
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "JOBS", jobs)
    monkeypatch.setattr(vl, "ready", lambda: True)
    s = server.Server(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield s
    s.shutdown()
    s.server_close()
    jobs.pool.shutdown(wait=True, cancel_futures=True)


def _req(s, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", s.server_address[1], timeout=10)
    headers, data = {}, None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    try:
        c.request(method, path, body=data, headers=headers)
        r = c.getresponse()
        return r.status, json.loads(r.read())
    finally:
        c.close()


@pytest.mark.parametrize(
    "path, name",
    [("/api/day", "day"), ("/api/event", "ei"), ("/api/cc/window", "seg_id"), ("/api/agent_series", "agent")]
    + [("/api/job", "id"), ("/api/day?day=", "day")],
)
def test_missing_parameter_is_400(srv, path, name):
    assert _req(srv, "GET", path) == (400, {"error": f"missing parameter: {name}"})


def test_non_integer_parameter_is_400(srv):
    assert _req(srv, "GET", "/api/day?day=abc") == (400, {"error": "day must be an integer"})
    assert _req(srv, "GET", "/api/probes?limit=x") == (400, {"error": "limit must be an integer"})


def test_unknown_job_is_404(srv, monkeypatch):
    assert _req(srv, "GET", "/api/job?id=nope") == (404, {"error": "unknown job"})
    monkeypatch.setattr(server.JOBS, "get", lambda job_id: None)  # Jobs.get returning None for an unknown id
    assert _req(srv, "GET", "/api/job?id=nope") == (404, {"error": "unknown job"})

    def raises(job_id):
        raise KeyError(job_id)

    monkeypatch.setattr(server.JOBS, "get", raises)
    assert _req(srv, "GET", "/api/job?id=nope") == (404, {"error": "unknown job"})


def test_known_job_is_200(srv):
    code, sub = _req(srv, "POST", "/api/probe", {"target": "tierb", "agent": "a", "ei": 1, "window": "80"})
    assert code == 202
    for _ in range(100):
        code, job = _req(srv, "GET", f"/api/job?id={sub['job_id']}")
        if job.get("status") == "done":
            break
        time.sleep(0.02)
    assert code == 200 and job["job_id"] == sub["job_id"] and job["result"]["text"] == "answer"


@pytest.mark.parametrize(
    "path, body, error",
    [
        ("/api/probe", {"target": 5}, "target must be a string"),
        ("/api/probe", {"mode": ["belief"], "ei": 1}, "mode must be a string"),
        ("/api/probe", {"parent_id": [1]}, "parent_id must be a string"),
        ("/api/probe", {"ei": 1, "agent": ["x"]}, "agent must be a string"),
        ("/api/probe", {"ei": 1, "agent": "a", "question": 5}, "question must be a string"),
        ("/api/probe", {"ei": 1, "agent": "a", "model": 5}, "model must be a string"),
        ("/api/probe", {"ei": [1]}, "ei must be an integer"),
        ("/api/probe", {"ei": True}, "ei must be an integer"),
        ("/api/probe", {"ei": 1, "window": 2.5}, "window must be an integer"),
        ("/api/stars", {"op": "add", "probe_id": {"a": 1}}, "probe_id must be a string"),
        ("/api/stars", {"op": "add", "ei": [1]}, "ei must be an integer"),
        ("/api/stars", {"op": ["add"]}, "op must be a string"),
        ("/api/stars", {"op": "remove", "id": 3}, "id must be a string"),
    ],
)
def test_wrongly_typed_json_fields_are_400(srv, capsys, path, body, error):
    assert _req(srv, "POST", path, body) == (400, {"error": error})
    assert "Traceback" not in capsys.readouterr().err
    assert not server.JOBS.jobs


def test_internal_lookup_error_is_500_not_400(srv, monkeypatch, capsys):
    def broken(ei):
        raise KeyError("seg_turns")  # a bug inside the index, not a bad request

    monkeypatch.setattr(vl, "event", broken)
    internal = (500, {"error": "internal error (see server log)"})
    assert _req(srv, "GET", "/api/event?ei=1") == internal
    assert "KeyError: 'seg_turns'" in capsys.readouterr().err
    assert _req(srv, "POST", "/api/stars", {"op": "add", "ei": 1}) == internal
    assert "KeyError: 'seg_turns'" in capsys.readouterr().err
    assert not server.STORE.stars


def test_warming_up_is_503(srv, monkeypatch):
    monkeypatch.setattr(vl, "ready", lambda: False)
    code, body = _req(srv, "GET", "/api/day?day=1")
    assert code == 503 and body["loading"] is True
    assert _req(srv, "GET", "/api/job?id=x")[0] == 503
    assert _req(srv, "POST", "/api/probe", {"ei": 1, "agent": "a"})[0] == 503
    assert _req(srv, "GET", "/api/models")[0] == 200  # the picker works while the index loads


def test_failed_load_says_why(srv, monkeypatch):
    monkeypatch.setattr(vl, "ready", lambda: False)
    monkeypatch.setattr(vl, "status", lambda: {"ready": False, "stage": "error", "elapsed_s": 1.0, "error": "boom"})
    code, body = _req(srv, "GET", "/api/day?day=1")
    assert (
        code == 503 and body["loading"] is False and body["error"] == "server error: boom" and body["detail"] == "boom"
    )


def test_second_server_cannot_bind_a_busy_port():
    a = server.Server(("127.0.0.1", 0), server.Handler)
    try:
        with pytest.raises(OSError):
            server.Server(("127.0.0.1", a.server_address[1]), server.Handler)
    finally:
        a.server_close()


def test_no_reuse_address_on_windows():
    # on Windows SO_REUSEADDR lets a second server bind a port that is already serving: re-run the class
    # body as it would evaluate there
    assert server.Server.allow_reuse_address is (os.name != "nt")
    ns = {"os": types.SimpleNamespace(name="nt"), "socketserver": socketserver, "http": http}
    exec(inspect.getsource(server.Server), ns)
    assert ns["Server"].allow_reuse_address is False


def test_ctrl_c_cancels_queued_probes_and_waits_for_running(tmp_path, monkeypatch, capsys):
    started, release = [], threading.Event()

    def call_fn(system, messages, model):
        started.append(time.time())
        release.wait(5)
        return {"text": "done"}

    def boot():
        server.STORE = pj.Store(tmp_path)
        server.JOBS = pj.Jobs(server.STORE, _pp, call_fn, workers=1)

    def serve_forever(self, *a, **kw):
        for _ in range(3):
            server.JOBS.submit({})
        while not started:
            time.sleep(0.01)
        threading.Timer(0.3, release.set).start()
        raise KeyboardInterrupt

    monkeypatch.setattr(server, "STORE", None)
    monkeypatch.setattr(server, "JOBS", None)
    monkeypatch.setattr(server, "_boot", boot)
    monkeypatch.setattr(server.Server, "serve_forever", serve_forever)
    monkeypatch.setattr(sys, "argv", ["village-introspect", "--port", "0"])
    t0 = time.time()
    server.main()
    assert time.time() - t0 < 3
    server.JOBS.pool.shutdown(wait=True)
    time.sleep(0.3)  # time for a queued call to start, if shutdown failed to cancel it
    assert len(started) == 1  # the two queued probes were never sent
    assert [r["status"] for r in server.STORE.probes.values()] == ["done"]  # the running one was logged
    err = capsys.readouterr().err
    assert "2 queued probe(s) cancelled" in err and "waiting for 1 running model call(s)" in err

"""server.py — the local server behind the AI Village introspection viewer (timeline + ask).

It makes LIVE model calls with your API keys, so it runs locally rather than as a static page. The HTTP
server starts immediately; the corpora load in a background thread (Claude Code rows ~15s -> village
chat/activity ~15-90s -> timeline index ~15s), with progress on /api/status. Timeline routes return 503
until the index is ready.

    village-introspect --port 8765          # or: python -m village_introspect --port 8765

Configuration is by environment variable (see config.py and llm.py): VILLAGE_DATASET (required),
VILLAGE_STATE, VILLAGE_MODELS, ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY.

Routes (same-origin; POST requires application/json):
    GET  /                                   -> the timeline viewer (viewer.html)
    GET  /api/status                         -> {ready, stage, elapsed_s, error?}
    GET  /api/overview                       -> days/eras/agents/rooms/presence/moments + stars/probes by day
    GET  /api/day?day=N[&full=1]             -> every event of the day (light; full=1: complete text) + cc_segments + talked_about
    GET  /api/event?ei=N                     -> one event in full (+ thinking, detail, probe routing)
    GET  /api/search?q&days=a-b&limit&offset -> grammar search, per-day histogram, jump
    GET  /api/agent_series?agent=slug        -> own msgs/day + mentioned-by-others/day (approx)
    GET  /api/cc/window?seg_id=N             -> one Claude Code window as turns
    GET  /api/probe/prompt?…                 -> the exact prompt a probe would send + fidelity + actual
    POST /api/probe {…params, model} | {parent_id, question} -> {job_id, probe_id, status}
    GET  /api/job?id= · /api/jobs            -> job status (+ result when done) · recent jobs
    GET  /api/probes?day&ei&agent&seg_id&limit · /api/probes/<id>   -> probe log summaries · full record
    GET/POST /api/stars                      -> stars
    GET  /api/models                         -> probe models (usable = its provider's API key is set)

Fidelity: the Claude Code agent's replay is near-faithful (its real request+response turns; the exact
Claude Code system prompt and any pre-window compaction summary are missing); every other agent is an
approximate observable-chat reconstruction. Every probe response carries its fidelity block.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import traceback
import urllib.parse
from pathlib import Path

from . import config, llm
from . import probe_jobs as pj
from . import tierb_lib as tb
from . import village_lib as vl

HERE = Path(__file__).resolve().parent
HTML_PATH = HERE / "viewer.html"
MAX_BODY = 4_000_000
_OK_HOSTS = ("127.0.0.1", "localhost", "::1", "")
DAY_CACHE_N = 48

STORE: "pj.Store | None" = None
JOBS: "pj.Jobs | None" = None
_DAY_CACHE: "collections.OrderedDict" = collections.OrderedDict()
_DAY_LOCK = threading.Lock()

MODELS: list = []


def _models() -> list:
    """The probe model picker: the configured ids, each usable iff its provider's API key is set."""
    out = []
    for mid, note in config.configured_models():
        prov = llm.provider_for(mid)
        if prov is None:  # e.g. a bare grok-4 in VILLAGE_MODELS: shown, but it can't be asked
            hint = "unknown provider: use its OpenRouter id (vendor/model)"
            out.append({"id": mid, "usable": False, "note": " · ".join(x for x in (hint, note) if x), "provider": None})
            continue
        usable = llm.has_key(prov)
        parts = [{"openrouter": "via OpenRouter", "openai": "via OpenAI"}.get(prov, ""), note]
        if not usable:
            parts.insert(0, f"set {llm.PROVIDER_KEYS[prov]} to enable")
        note = " · ".join(x for x in parts if x)
        out.append({"id": mid, "usable": usable, "note": note, "provider": prov})
    return out


TIMELINE_ROUTES = (
    "/api/overview",
    "/api/day",
    "/api/event",
    "/api/search",
    "/api/agent_series",
    "/api/cc/window",
    "/api/probe/prompt",
    "/api/probe",
    "/api/job",
    "/api/jobs",
    "/api/probes",
    "/api/stars",
)


def _boot() -> None:
    global STORE, JOBS, MODELS
    if not os.environ.get("VILLAGE_DATASET"):
        sys.exit(
            "VILLAGE_DATASET is not set. Point it at the directory holding the AI Village dataset files "
            f"({', '.join(config.REQUIRED_FILES)}); see the README."
        )
    missing = config.missing_files()
    if missing:
        sys.exit(f"AI Village dataset incomplete at {config.DATASET}: missing {missing}.")
    MODELS = _models()
    usable = [m["id"] for m in MODELS if m["usable"]]
    pref = config.preferred_default()
    vl.DEFAULT_MODEL = pref if pref in usable else (usable or [MODELS[0]["id"]])[0]
    keys = [var for prov, var in llm.PROVIDER_KEYS.items() if llm.has_key(prov)]
    print(f"  dataset: {config.DATASET}\n  state:   {config.STATE_DIR}", file=sys.stderr)
    print(f"  API keys found: {', '.join(keys) or 'none'}; default probe model: {vl.DEFAULT_MODEL}", file=sys.stderr)
    if not usable:
        print(
            "  WARNING: no probe model is usable, so asking will fail. Set ANTHROPIC_API_KEY, OPENAI_API_KEY or "
            "OPENROUTER_API_KEY (browsing works without one).",
            file=sys.stderr,
        )
    STORE = pj.Store(config.STATE_DIR)
    JOBS = pj.Jobs(STORE, vl.probe_prompt, vl.call_model, workers=_WORKERS)
    print(f"  probe log: {len(STORE.probes)} probes, {len(STORE.stars)} stars", file=sys.stderr)

    def _cc_loaded(rows):
        print(f"  [bg] Claude Code corpus: {len(rows)} rows", file=sys.stderr)

    def _bg():
        try:
            vl.warm(_cc_loaded)
            print(
                f"  [bg] village index ready ({len(vl.overview()['days'])} days, "
                f"{len(tb.load_chat())} chat, {len(tb.load_activity())} activity) in "
                f"{vl.status()['elapsed_s']}s",
                file=sys.stderr,
            )
        except Exception:
            traceback.print_exc()

    threading.Thread(target=_bg, daemon=True).start()


def _qs(q: dict) -> dict:
    return {k: v[0] for k, v in q.items() if v}


def _int_or_none(v):
    return int(v) if v not in (None, "") else None


_REQUIRED = object()


def _arg(q: dict, name: str, default=_REQUIRED):
    """A query parameter (parse_qs output). A missing required one is a 400 "missing parameter: <name>"."""
    v = q.get(name, [""])[0]
    if v != "":
        return v
    if default is _REQUIRED:
        raise ValueError(f"missing parameter: {name}")
    return default


def _int_arg(q: dict, name: str, default=_REQUIRED):
    v = _arg(q, name, default)
    if not isinstance(v, str):
        return v  # the default
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None


# JSON body fields that must be strings / integers when present (integers may also come as digit strings,
# as in a query string); anything else is a 400 rather than a crash deeper in.
_STR_FIELDS = ("target", "mode", "anchor", "format", "agent", "question", "model", "parent_id", "probe_id", "op", "id")
_INT_FIELDS = ("ei", "seg_id", "seq", "at_ei", "window")


def _check_fields(req: dict) -> None:
    for k in _STR_FIELDS:
        if req.get(k) is not None and not isinstance(req[k], str):
            raise ValueError(f"{k} must be a string")
    for k in _INT_FIELDS:
        v = req.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, str))):
            raise ValueError(f"{k} must be an integer")


def _day_bytes(n: int, full: bool = False) -> bytes:
    """Serialized /api/day with probe/star annotations; LRU keyed by (day, full, store version)."""
    key = (n, full, STORE.version)
    with _DAY_LOCK:
        if key in _DAY_CACHE:
            _DAY_CACHE.move_to_end(key)
            return _DAY_CACHE[key]
    d = vl.day(n, full=full)
    nprobes, stars = STORE.n_probes_by_ei(), STORE.starred_eis()
    evs = []
    for e in d["events"]:
        if e["k"] == "c":
            e = dict(e, n_probes=nprobes.get(e["ei"], 0), star=e["ei"] in stars)
        evs.append(e)
    body = json.dumps(dict(d, events=evs)).encode()
    with _DAY_LOCK:
        _DAY_CACHE[key] = body
        while len(_DAY_CACHE) > DAY_CACHE_N:
            _DAY_CACHE.popitem(last=False)
    return body


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30

    def log_message(self, fmt, *args):
        sys.stderr.write("  [srv] " + (fmt % args) + "\n")

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in _OK_HOSTS

    def _send(self, code, body: bytes, ctype: str):
        gz = len(body) > 32_000 and "gzip" in (self.headers.get("Accept-Encoding") or "")
        if gz:  # a busy day's event stream is ~1.4MB raw
            body = gzip.compress(body, compresslevel=5)
        self.send_response(code)
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def _err(self, code, msg, **extra):
        self._json(code, {"error": msg, **extra})

    def _loading(self):
        st = vl.status()
        if st["stage"] == "error":  # the load failed: say why instead of "still loading"
            why = st.get("error") or "the village index failed to load"
            return self._err(503, f"server error: {why}", loading=False, stage="error", detail=why)
        return self._err(
            503,
            "warming up: the village index is still loading",
            loading=True,
            stage=st["stage"],
            elapsed_s=st["elapsed_s"],
            **({"detail": st["error"]} if st.get("error") else {}),
        )

    # ------------------------------------------------------------------------------------------ GET
    def do_GET(self):
        if not self._host_ok():
            return self._err(403, "forbidden host")
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        path = u.path
        try:
            if path in ("/", "/index.html"):
                return self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._json(200, vl.status())
            if (path in TIMELINE_ROUTES or path.startswith("/api/probes/")) and not vl.ready():
                return self._loading()
            if path == "/api/models":
                return self._json(200, {"default": vl.DEFAULT_MODEL, "models": MODELS})
            if path == "/api/overview":
                ov = dict(vl.overview())
                ov["stars_by_day"] = STORE.stars_by_day()
                ov["probes_by_day"] = STORE.probes_by_day()
                return self._json(200, ov)
            if path == "/api/day":
                body = _day_bytes(_int_arg(q, "day"), _arg(q, "full", "0") == "1")
                return self._send(200, body, "application/json; charset=utf-8")
            if path == "/api/event":
                e = vl.event(_int_arg(q, "ei"))
                if e["k"] == "c":
                    e["n_probes"] = STORE.n_probes_by_ei().get(e["ei"], 0)
                    e["star"] = e["ei"] in STORE.starred_eis()
                return self._json(200, e)
            if path == "/api/search":
                days = None
                if q.get("days", [""])[0]:
                    a, _, b = q["days"][0].partition("-")
                    days = (int(a), int(b or a))
                return self._json(
                    200,
                    vl.search(
                        q.get("q", [""])[0],
                        days=days,
                        limit=max(1, min(_int_arg(q, "limit", 500), 5000)),
                        offset=max(0, _int_arg(q, "offset", 0)),
                    ),
                )
            if path == "/api/agent_series":
                return self._json(200, vl.agent_series(_arg(q, "agent")))
            if path == "/api/cc/window":
                return self._json(200, vl.cc_window(_int_arg(q, "seg_id")))
            if path == "/api/probe/prompt":
                return self._json(200, vl.probe_prompt(_qs(q)))
            if path == "/api/job":
                job_id = _arg(q, "id")
                try:
                    job = JOBS.get(job_id)
                except (LookupError, ValueError):  # an unknown id, if Jobs.get raises rather than returns None
                    job = None
                return self._json(200, job) if job is not None else self._err(404, "unknown job")
            if path == "/api/jobs":
                return self._json(200, JOBS.list())
            if path == "/api/probes":
                return self._json(
                    200,
                    STORE.list(
                        day=_int_arg(q, "day", None),
                        ei=_int_arg(q, "ei", None),
                        agent=_arg(q, "agent", None),
                        seg_id=_int_arg(q, "seg_id", None),
                        limit=min(_int_arg(q, "limit", 100), 1000),
                    ),
                )
            if path.startswith("/api/probes/"):
                rec = STORE.get(path.rsplit("/", 1)[1])
                return self._json(200, rec) if rec else self._err(404, "unknown probe_id")
            if path == "/api/stars":
                return self._json(200, STORE.star_list())
            return self._err(404, "not found")
        except ValueError as e:  # bad or missing parameters, unknown ids
            return self._err(400, str(e))
        except Exception:
            traceback.print_exc()
            return self._err(500, "internal error (see server log)")

    # ----------------------------------------------------------------------------------------- POST
    def do_POST(self):
        if not self._host_ok():
            return self._err(403, "forbidden host")
        u = urllib.parse.urlparse(self.path)
        if u.path not in ("/api/probe", "/api/stars"):
            return self._err(404, "not found")
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self._err(415, "expected application/json")
        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return self._err(400, "bad Content-Length")
        if n <= 0 or n > MAX_BODY:
            return self._err(413, "missing or oversized body")
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(req, dict):
                return self._err(400, "body must be a JSON object")
            _check_fields(req)
            if not vl.ready():
                return self._loading()
            if u.path == "/api/probe":
                return self._json(202, JOBS.submit(req))
            if u.path == "/api/stars":
                op = req.get("op")
                if op == "add":
                    ei = _int_or_none(req.get("ei"))
                    day = None
                    if ei is not None:
                        day = vl.event(ei)["day"]
                    elif req.get("probe_id"):
                        rec = STORE.get(req["probe_id"])
                        if rec is None:
                            return self._err(400, "unknown probe_id")
                        day = rec.get("day")
                    return self._json(
                        200,
                        STORE.star_add(
                            ei=ei, probe_id=req.get("probe_id"), day=day, note=str(req.get("note") or "")[:500]
                        ),
                    )
                if op == "remove":
                    k = STORE.star_remove(
                        id=req.get("id"), ei=_int_or_none(req.get("ei")), probe_id=req.get("probe_id")
                    )
                    return self._json(200, {"removed": k})
                return self._err(400, "op must be 'add' or 'remove'")
        except ValueError as e:  # bad fields, unknown ids
            return self._err(400, str(e))
        except Exception:
            traceback.print_exc()
            return self._err(500, "internal error (see server log)")


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    # SO_REUSEADDR lets a restart rebind a port in TIME_WAIT. On Windows it also lets a second server bind a
    # port that is already serving, so it stays off there (HTTPServer turns it on by default).
    allow_reuse_address = os.name != "nt"


_WORKERS = pj.MAX_WORKERS


def _stop_jobs() -> None:
    """On Ctrl-C: cancel queued probes, so they are never sent (or billed), and wait for the running calls
    so their answers are logged; a second Ctrl-C quits at once and abandons them."""
    if JOBS is None:
        return

    def count(status):
        return sum(j["status"] == status for j in list(JOBS.jobs.values()))

    queued = count("queued")
    JOBS.pool.shutdown(wait=False, cancel_futures=True)
    if queued:
        print(f"  stopping: {queued} queued probe(s) cancelled, not sent", file=sys.stderr)
    if not count("running"):
        return
    print(
        f"  waiting for {count('running')} running model call(s) to finish and be logged; Ctrl-C again to quit now",
        file=sys.stderr,
    )
    try:
        while count("running"):
            time.sleep(0.2)
    except KeyboardInterrupt:
        print(f"  quitting: {count('running')} running call(s) abandoned, not logged", file=sys.stderr, flush=True)
        os._exit(130)  # the pool's worker threads would otherwise be joined at interpreter exit


def main():
    global _WORKERS
    ap = argparse.ArgumentParser(
        prog="village-introspect", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind (default: localhost only)")
    ap.add_argument("--workers", type=int, default=pj.MAX_WORKERS, help="concurrent live model calls")
    args = ap.parse_args()
    _WORKERS = max(1, args.workers)
    try:
        srv = Server((args.host, args.port), Handler)  # bind first, so a busy port fails before loading
    except OSError as e:
        sys.exit(f"can't listen on {args.host}:{args.port}: {e.strerror}. Try another --port.")
    try:
        _boot()
        print(f"  serving http://{args.host}:{args.port}/  (Ctrl-C to stop)", file=sys.stderr)
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.server_close()
        _stop_jobs()


if __name__ == "__main__":
    main()

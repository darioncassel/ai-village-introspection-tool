"""probe_jobs.py — server-side probe jobs, the append-only probe log, and stars.

A probe is (1) resolved to its exact prompt synchronously (village_lib.probe_prompt — bad params fail
fast with a 400), then (2) queued on a small thread pool that makes the live call, so the viewer can
keep browsing, close the drawer, or reload; jobs still finish and are logged. Every finished probe
(done OR error) is appended to <state>/probes.jsonl with the FULL system + messages it sent
(follow-ups included) and their sha256 — the permanent, replayable record ("prompt used").

Follow-ups: the client sends {parent_id, question}; the thread is rebuilt from the stored chain:
messages = parent.messages + [assistant: parent.text] + [user: question], inheriting the root's
resolved anchor / mode / model / window / fidelity. Stars live in <state>/stars.jsonl (op log).
"""

from __future__ import annotations

import collections
import datetime
import json
import os
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

MAX_WORKERS = 4  # concurrent live calls; the rest queue
JOB_TTL_S = 30 * 60  # /api/jobs lists jobs from the last 30 min (plus anything still in flight)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _read_log(path: Path):
    """Yield the JSON objects of an append-only log, one per line. A line that can't be used (torn by a
    crash or a full disk, blank, not UTF-8, not JSON, not an object) is skipped with a warning on stderr, so
    one bad line never stops the server from starting. Only "\n" ends a record: the logs are written
    with ensure_ascii=False, so U+2028 / U+2029 / U+0085 appear raw inside records (str.splitlines()
    would cut them apart)."""
    if not path.exists():
        return
    with open(path, "rb") as f:  # binary iteration splits on b"\n" only; decode each line separately
        for n, raw in enumerate(f, 1):
            if not raw.strip():
                _skip(path, n, "blank line")
                continue
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (ValueError, RecursionError) as e:  # UnicodeDecodeError is a ValueError too
                if not raw.endswith(b"\n"):
                    _skip(path, n, "incomplete last line (a write that was cut short?)")
                else:
                    _skip(path, n, "not valid UTF-8" if isinstance(e, UnicodeDecodeError) else "not valid JSON")
                continue
            if isinstance(obj, dict):
                yield n, obj
            else:
                _skip(path, n, "not a JSON object")


def _skip(path: Path, n: int, why: str) -> None:
    print(f"warning: {path} line {n}: {why}; skipped", file=sys.stderr)


def _append_line(path: Path, obj) -> None:
    """Append one JSON line. Serialised before the file is touched, and started on a fresh line if the
    file doesn't end with one (a torn last line would otherwise swallow this record)."""
    data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    with open(path, "a+b") as f:
        if f.seek(0, os.SEEK_END) > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                data = b"\n" + data
        f.write(data)


class Store:
    """probes.jsonl + stars.jsonl, loaded at start, appended as things happen. Every change is written
    to disk first and applied in memory only once the write succeeded. `lock` guards the in-memory
    dicts (readers snapshot under it); `write_lock` serialises changes, so readers never wait on disk."""

    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.probes_path = self.dir / "probes.jsonl"
        self.stars_path = self.dir / "stars.jsonl"
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.version = 0  # bumped on every change (day-cache key)
        self.probes: dict = {}
        self.stars: dict = {}
        for n, rec in _read_log(self.probes_path):
            if isinstance(rec.get("probe_id"), str) and rec["probe_id"]:
                self.probes[rec["probe_id"]] = rec
            else:
                _skip(self.probes_path, n, "no probe_id")
        for n, op in _read_log(self.stars_path):
            sid = op.get("id")
            if not (isinstance(sid, str) and sid) or op.get("op") not in ("add", "remove"):
                _skip(self.stars_path, n, "not a star add/remove")
            elif op["op"] == "add":
                self.stars[sid] = {k: op.get(k) for k in ("id", "ei", "probe_id", "day", "note", "created_at")}
            else:
                self.stars.pop(sid, None)

    # ---- probes ----
    def append_probe(self, rec: dict) -> None:
        """Raises (and changes nothing in memory) if the record can't be written."""
        with self.write_lock:
            _append_line(self.probes_path, rec)
            with self.lock:
                self.probes[rec["probe_id"]] = rec
                self.version += 1

    def get(self, probe_id: str) -> Optional[dict]:
        with self.lock:
            return self.probes.get(probe_id)

    @staticmethod
    def summary(rec: dict) -> dict:
        return {
            "probe_id": rec["probe_id"],
            "parent_id": rec.get("parent_id"),
            "root_id": rec.get("root_id"),
            "anchor": rec.get("anchor"),
            "resolved": rec.get("resolved"),
            "day": rec.get("day"),
            "agent": rec.get("agent"),
            "mode": rec.get("mode"),
            "question": rec.get("question"),
            "model": rec.get("model"),
            "fidelity_level": (rec.get("fidelity") or {}).get("level"),
            "status": rec.get("status"),
            "created_at": rec.get("created_at"),
            "text_preview": (rec.get("text") or rec.get("error") or "")[:200],
        }

    def _probe_values(self) -> list:
        with self.lock:
            return list(self.probes.values())

    def _star_values(self) -> list:
        with self.lock:
            return list(self.stars.values())

    def list(self, *, day=None, ei=None, agent=None, seg_id=None, limit=100) -> list:
        out = []
        for rec in sorted(self._probe_values(), key=lambda r: r.get("created_at") or "", reverse=True):
            res = rec.get("resolved") or {}
            if day is not None and rec.get("day") != day:
                continue
            if ei is not None and res.get("ei") != ei:
                continue
            if agent and agent not in (rec.get("agent"), rec.get("agent_id")):
                continue
            if seg_id is not None and res.get("seg_id") != seg_id:
                continue
            out.append(self.summary(rec))
            if len(out) >= limit:
                break
        return out

    def _roots_done(self):
        return [r for r in self._probe_values() if not r.get("parent_id") and r.get("status") == "done"]

    def probes_by_day(self) -> dict:
        c = collections.Counter(r.get("day") for r in self._roots_done() if r.get("day") is not None)
        return dict(c)

    def n_probes_by_ei(self) -> dict:
        c = collections.Counter((r.get("resolved") or {}).get("ei") for r in self._roots_done())
        c.pop(None, None)
        return dict(c)

    # ---- stars ----
    def star_list(self) -> list:
        return sorted(self._star_values(), key=lambda s: s.get("created_at") or "")

    def stars_by_day(self) -> dict:
        return dict(collections.Counter(s.get("day") for s in self._star_values() if s.get("day") is not None))

    def starred_eis(self) -> set:
        return {s["ei"] for s in self._star_values() if s.get("ei") is not None}

    # star changes hold write_lock throughout, so self.stars can't change under their scans
    def star_add(self, *, ei=None, probe_id=None, day=None, note="") -> dict:
        if ei is None and not probe_id:
            raise ValueError("star needs ei or probe_id")
        with self.write_lock:
            for s in self.stars.values():  # idempotent per target
                if s.get("ei") == ei and s.get("probe_id") == probe_id:
                    return s
            s = {
                "id": uuid.uuid4().hex[:10],
                "ei": ei,
                "probe_id": probe_id,
                "day": day,
                "note": note or "",
                "created_at": _now_iso(),
            }
            _append_line(self.stars_path, {"op": "add", **s})
            with self.lock:
                self.stars[s["id"]] = s
                self.version += 1
            return s

    def star_remove(self, *, id=None, ei=None, probe_id=None) -> int:
        with self.write_lock:
            ids = [
                k
                for k, s in self.stars.items()
                if (id and k == id)
                or (
                    id is None
                    and ((ei is not None and s.get("ei") == ei) or (probe_id and s.get("probe_id") == probe_id))
                )
            ]
            if not ids:
                raise ValueError("no such star")
            for k in ids:
                _append_line(self.stars_path, {"op": "remove", "id": k, "created_at": _now_iso()})
                with self.lock:
                    self.stars.pop(k, None)
                    self.version += 1
            return len(ids)


class Jobs:
    """A job = one live model call for a (resolved) probe. prompt_fn(params) -> probe_prompt dict;
    call_fn(system, messages, model) -> {thinking, text, tool_use, usage, stop_reason, request}.
    A job turns done/error only once its record is in the Store (or saving it has failed)."""

    def __init__(self, store: Store, prompt_fn: Callable, call_fn: Callable, workers: int = MAX_WORKERS):
        self.store, self.prompt_fn, self.call_fn = store, prompt_fn, call_fn
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="probe")
        self.jobs: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
        self.lock = threading.Lock()

    def _new_record(self, pp: dict, params: dict, question: str, parent: Optional[dict]) -> dict:
        res = pp["resolved"]
        return {
            "probe_id": uuid.uuid4().hex[:12],
            "parent_id": parent["probe_id"] if parent else None,
            "root_id": (parent.get("root_id") or parent["probe_id"]) if parent else None,
            "created_at": _now_iso(),
            "status": "queued",
            "target": res["target"],
            "agent": res["agent"],
            "agent_id": res["agent_id"],
            "agent_name": res.get("agent_name"),
            "resolved": res,
            "day": res.get("day"),
            "room": res.get("room"),
            "mode": pp["mode"],
            "question": question,
            "question_effective": question.strip() or pp.get("default_question") or "",
            "model": pp["model"],
            "window": pp.get("window"),
            "format": pp.get("format"),
            "anchor": res.get("anchor"),
            "anchor_desc": pp.get("anchor_desc"),
            "context": pp.get("context"),
            "fidelity": pp["fidelity"],
            "system": pp["system"],
            "messages": pp["messages"],
            "sha256": pp["sha256"],
            "thinking": "",
            "text": "",
            "tool_use": None,
            "usage": None,
            "stop_reason": None,
            "request": None,
            "actual": pp.get("actual"),
            "elapsed_s": None,
            "error": None,
            "params": params,
        }

    def submit(self, body: dict) -> dict:
        """body = probe params (+model)  or  {parent_id, question}. Raises ValueError on bad input."""
        if body.get("parent_id"):
            pid = body["parent_id"]
            parent = self.store.get(pid)
            if parent is None:  # not in the log: still being answered, unsaved, or unknown
                with self.lock:
                    pending = next((j["status"] for j in self.jobs.values() if j["probe_id"] == pid), None)
                parent = self.store.get(pid)  # it may have been saved in between
                if parent is None and pending in ("queued", "running"):
                    raise ValueError("wait for the previous answer to finish, then follow up")
                if parent is None and pending is not None:
                    raise ValueError("can't follow up on that answer: it wasn't saved to the probe log")
                if parent is None:
                    raise ValueError("unknown parent_id")
            if parent.get("status") != "done":
                raise ValueError("can only follow up on a finished probe")
            q = (body.get("question") or "").strip()
            if not q:
                raise ValueError("a follow-up needs a question")
            from .village_lib import prompt_sha  # local import: keep this module stdlib-only at top

            msgs = list(parent["messages"]) + [
                {"t": "assistant", "text": parent.get("text") or ""},
                {"t": "user", "text": q},
            ]
            pp = {
                "resolved": parent["resolved"],
                "mode": parent["mode"],
                "model": parent["model"],
                "window": parent.get("window"),
                "format": parent.get("format"),
                "anchor_desc": parent.get("anchor_desc"),
                "context": parent.get("context"),
                "fidelity": parent["fidelity"],
                "system": parent["system"],
                "messages": msgs,
                "sha256": prompt_sha(parent["system"], msgs),
                "actual": parent.get("actual"),
                "default_question": "",
            }
            rec = self._new_record(pp, parent.get("params") or {}, q, parent)
        else:
            pp = self.prompt_fn(body)
            rec = self._new_record(
                pp, {k: v for k, v in body.items() if k != "followups"}, body.get("question") or "", None
            )
        job = {
            "job_id": uuid.uuid4().hex[:12],
            "probe_id": rec["probe_id"],
            "status": "queued",
            "t_submit": time.time(),
            "t_start": None,
            "t_end": None,
            "record": rec,
            "error": None,
        }
        with self.lock:
            self.jobs[job["job_id"]] = job
            self._prune()
        self.pool.submit(self._run, job)
        return {"job_id": job["job_id"], "probe_id": rec["probe_id"], "status": "queued"}

    def _run(self, job: dict) -> None:
        rec = job["record"]
        with self.lock:
            job["status"] = rec["status"] = "running"
            job["t_start"] = time.time()
        try:
            out = self.call_fn(rec["system"], rec["messages"], rec["model"])
            rec.update(
                thinking=out.get("thinking") or "",
                text=out.get("text") or "",
                tool_use=out.get("tool_use"),
                usage=out.get("usage"),
                stop_reason=out.get("stop_reason"),
                request=out.get("request"),
                status="done",
            )
        except Exception as e:  # recorded, not raised: the client polls
            rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:500]}")
        t_end = time.time()
        rec["elapsed_s"] = round(t_end - job["t_start"], 1)
        status, error = rec["status"], rec["error"]
        # Saved outside the except above, so a save error's traceback never chains the call's exception.
        try:
            self.store.append_probe(rec)
        except Exception as e:
            print(f"probe {rec['probe_id']}: couldn't append to {self.store.probes_path}:", file=sys.stderr)
            traceback.print_exc()
            why = f"couldn't be saved to the probe log: {type(e).__name__}: {str(e)[:500]}"
            rec["save_error"] = why  # in memory only: the result itself says why it won't survive a restart
            status, error = "error", (f"answer received but {why}" if status == "done" else f"{error} (and it {why})")
        with self.lock:  # published last: a client that sees done/error can rely on the Store
            job.update(t_end=t_end, error=error, status=status)

    def _prune(self) -> None:
        cut = time.time() - JOB_TTL_S
        for k in [k for k, j in self.jobs.items() if j["t_end"] and j["t_end"] < cut]:
            self.jobs.pop(k, None)

    def _queue_pos(self, job: dict) -> Optional[int]:
        if job["status"] != "queued":
            return None
        q = [j for j in self.jobs.values() if j["status"] == "queued"]
        return next((i + 1 for i, j in enumerate(q) if j is job), None)

    def view(self, job: dict, with_result: bool = True) -> dict:
        """Call with self.lock held (the queue position scans self.jobs)."""
        rec = job["record"]
        ref = job["t_end"] or time.time()
        v = {
            "job_id": job["job_id"],
            "probe_id": job["probe_id"],
            "status": job["status"],
            "elapsed_s": round(ref - (job["t_start"] or job["t_submit"]), 1),
            "queue_pos": self._queue_pos(job),
            "agent": rec.get("agent"),
            "day": rec.get("day"),
            "ei": (rec.get("resolved") or {}).get("ei"),
            "question": rec.get("question"),
            "parent_id": rec.get("parent_id"),
        }
        if job["error"]:
            v["error"] = job["error"]
        if with_result and job["status"] in ("done", "error"):
            v["result"] = rec
        return v

    def get(self, job_id: str) -> Optional[dict]:
        """The job's view, or None for an unknown (or pruned) job id."""
        with self.lock:
            job = self.jobs.get(job_id)
            return None if job is None else self.view(job)

    def list(self) -> list:
        with self.lock:
            self._prune()
            return [self.view(j, with_result=False) for j in reversed(self.jobs.values())]

"""Where the tool finds the AI Village dataset, where it keeps its own state, and which probe models it
offers. Everything is configured through environment variables:

    VILLAGE_DATASET   directory holding the AI Village dataset files (required)
    VILLAGE_STATE     where the probe log, stars and caches go (default: ~/.village-introspect)
    VILLAGE_MODELS    comma-separated probe model ids, replacing the default list (first = default)

API keys are read by llm.py: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# the four dataset files the server reads (village-transcript.json only for the village day ranges)
REQUIRED_FILES = ("agents.jsonl.gz", "claude_code_messages.jsonl.gz", "events.jsonl.gz", "village-transcript.json")

DATASET = Path(os.environ.get("VILLAGE_DATASET") or ".").expanduser()
STATE_DIR = Path(os.environ.get("VILLAGE_STATE") or (Path.home() / ".village-introspect")).expanduser()

DEFAULT_MODEL = "claude-opus-4-8"
# (model id, note). The provider is inferred from the id (llm.provider_for): "vendor/model" ids go to
# OpenRouter, ids starting with "claude" to Anthropic, OpenAI's own ids (gpt-*, o*, ...) to OpenAI; any
# other bare id is refused.
DEFAULT_MODELS = (
    ("claude-opus-4-8", "default"),
    ("claude-opus-4-5-20251101", "era-matched for Opus 4.5 (Claude Code), the model that produced those turns"),
    ("claude-sonnet-4-6", ""),
    ("claude-haiku-4-5-20251001", "fast"),
    ("gpt-5.5", ""),
    ("google/gemini-3.1-pro-preview", ""),
    ("x-ai/grok-4.5", ""),
    ("deepseek/deepseek-v4-pro", ""),
    ("moonshotai/kimi-k3", ""),
)


def configured_models() -> list:
    """[(id, note)] from $VILLAGE_MODELS if set, else the defaults."""
    env = os.environ.get("VILLAGE_MODELS", "")
    ids = [m.strip() for m in env.split(",") if m.strip()]
    return [(m, "") for m in ids] if ids else list(DEFAULT_MODELS)


def preferred_default() -> str:
    """The first $VILLAGE_MODELS entry if set, else DEFAULT_MODEL (the server falls back to the first usable)."""
    return configured_models()[0][0] if os.environ.get("VILLAGE_MODELS", "").strip(", ") else DEFAULT_MODEL


def missing_files(dataset: Path = DATASET) -> list:
    return [f for f in REQUIRED_FILES if not (dataset / f).is_file()]


_TS = re.compile(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?\s*(Z|[+-]\d{2}:?\d{2})?")


def utc_seconds(ts) -> Optional[float]:
    """Epoch seconds for a dataset timestamp ('2026-06-15 16:00:00.861234' in the event tables, UTC;
    '2026-06-15T16:00:00.861Z' in the transcript). Any number of fraction digits; None if unparseable."""
    m = _TS.match(ts) if isinstance(ts, str) else None
    if not m:
        return None
    y, mo, d, h, mi, s, frac, tz = m.groups()
    try:
        t = datetime.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), tzinfo=datetime.timezone.utc)
    except ValueError:
        return None
    off = 0
    if tz and tz != "Z":
        hh, mm = int(tz[1:3]), int(tz[-2:])
        off = (hh * 3600 + mm * 60) * (1 if tz[0] == "+" else -1)
    return t.timestamp() - off + (int(frac[:6].ljust(6, "0")) / 1e6 if frac else 0.0)


_DAY_CACHE = "day_ranges.json"
_DAY_CACHE_VERSION = 1


def _ranges_from_transcript(t) -> list:
    """[{day, date, start, end}] (epoch seconds of the day's first and last event), sorted by start.
    Rows with no events are skipped (the transcript has a few empty stub days); malformed rows are
    skipped with a warning; no usable row at all is an error."""
    days = t.get("days") if isinstance(t, dict) else None
    if not isinstance(days, list):
        raise ValueError("it has no 'days' list")
    out, bad = [], 0
    for d in days:
        evs = d.get("events") if isinstance(d, dict) else None
        if isinstance(evs, list) and not evs:
            continue
        ts = [utc_seconds(e.get("timestamp")) for e in evs or () if isinstance(e, dict)]
        ts = [x for x in ts if x is not None]
        n = d.get("day") if isinstance(d, dict) else None
        if not ts or type(n) is not int:
            bad += 1
            continue
        date = d.get("date")
        if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            date = datetime.datetime.fromtimestamp(min(ts), datetime.timezone.utc).strftime("%Y-%m-%d")
        out.append({"day": n, "date": date, "start": min(ts), "end": max(ts)})
    if bad:
        print(f"  WARNING: skipped {bad} malformed day rows in village-transcript.json", file=sys.stderr)
    if not out:
        raise ValueError("it has no day with timestamped events")
    out.sort(key=lambda r: r["start"])
    return out


def _cached_ranges(c, stamp: dict) -> Optional[list]:
    """The ranges from a parsed cache file, or None if it is stale or not in the expected shape."""
    if not isinstance(c, dict) or c.get("version") != _DAY_CACHE_VERSION or c.get("stamp") != stamp:
        return None
    rs = c.get("days")
    if not isinstance(rs, list) or not rs:
        return None
    prev = None
    for r in rs:
        if not isinstance(r, dict) or type(r.get("day")) is not int or not isinstance(r.get("date"), str):
            return None
        a, b = r.get("start"), r.get("end")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b)) or a > b:
            return None
        if prev is not None and a < prev:
            return None
        prev = a
    return [{"day": r["day"], "date": r["date"], "start": float(r["start"]), "end": float(r["end"])} for r in rs]


def day_ranges(dataset: Path = DATASET) -> list:
    """The village days as time ranges from village-transcript.json, the dataset's own grouping of
    events into days: [{day, date, start, end}], start/end = epoch seconds of the day's first and last
    event, sorted by start. Days are assigned by time, not by calendar date, because a day's session
    can run past 00:00 UTC and two days can share a date.

    The transcript is ~360MB, so the ranges are cached under $VILLAGE_STATE/cache and rebuilt when the
    transcript's size or mtime changes; an unreadable or ill-shaped cache is rebuilt, and a cache that
    can't be written only costs the rebuild next time. Raises RuntimeError if the transcript is
    missing or can't be parsed (e.g. a truncated download)."""
    src = dataset / "village-transcript.json"
    try:
        st = src.stat()
    except OSError as e:
        raise RuntimeError(f"can't read {src}: {e.strerror or e}") from e
    stamp = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    cache = STATE_DIR / "cache" / _DAY_CACHE
    try:
        rs = _cached_ranges(json.loads(cache.read_text(encoding="utf-8")), stamp)
    except Exception:  # no cache yet, or unreadable: rebuild
        rs = None
    if rs is not None:
        return rs
    print("  building the village day index from village-transcript.json (one-time)...", file=sys.stderr)
    try:
        rs = _ranges_from_transcript(json.loads(src.read_text(encoding="utf-8")))
    except (OSError, ValueError) as e:  # ValueError covers bad JSON / encoding and a wrong shape
        raise RuntimeError(
            f"can't build the village days from {src}: {type(e).__name__}: {e} "
            "(if the file is incomplete, download it again)"
        ) from e
    tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"version": _DAY_CACHE_VERSION, "stamp": stamp, "days": rs}), encoding="utf-8")
        os.replace(tmp, cache)
    except OSError as e:
        print(f"  WARNING: could not cache the day index in {cache.parent} ({e}); continuing", file=sys.stderr)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return rs

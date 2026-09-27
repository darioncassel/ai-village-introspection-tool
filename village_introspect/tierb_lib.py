"""tierb_lib.py — approximate, observable-context reconstruction for every village agent except the
Claude Code one.

Routing: an agent is the Claude Code agent iff its `agents.model_string` starts with `claude-code::`
(exactly one agent, "Opus 4.5 (Claude Code)"); it gets the near-faithful replay in cc_lib. EVERY OTHER
agent is handled here, because the export has no `llm_calls` and its `computer_use_turns` / `events` are
output-only, so the exact prompt it received cannot be reconstructed. What CAN be reconstructed is:
  - the OBSERVABLE shared channel: the room chat (AGENT_TALK + USER_TALK) the agent could see up to a
    chosen message (a rolling window), and
  - the agent's OWN output at that message: its chat text and its raw logged thinking, in the shape its
    provider returned it: Anthropic thinking blocks and Gemini thought parts ('verbatim'), an OpenAI
    Responses reasoning SUMMARY ('summary'), or the chat-completions `reasoning` / `reasoning_content`
    text some providers return, e.g. DeepSeek, Grok, GLM, Kimi ('reasoning'). It is often absent for
    older and o-series models.

So this path is explicitly APPROXIMATE: no true system prompt, no private computer screen, no memory —
just the recent observable chat plus the agent's own words and thoughts. A live probe model (which may
differ from the agent's real model; flagged) answers in character, and the agent's ACTUAL logged
thinking and next message are shown for comparison.
"""

from __future__ import annotations

import bisect
import datetime
import gzip
import json
import re
import threading
from pathlib import Path
from typing import Optional

from .cc_lib import _BELIEF_PREFIX, DATASET, village_day

# The resample cue: an in-frame nudge to take the next turn. Unlike a belief question it has no
# _BELIEF_PREFIX, because the real agent's next turn was posted to the room, not kept private.
_RESAMPLE_Q = (
    "(Your turn: post your next message to the room now, or, if you'd rather act on your computer "
    "instead, say briefly what you'd do.)"
)
_BELIEF_Q = "What are you thinking about the situation right now, and what will you do next?"
# the idle-nudger's USER_TALK speaker name: a bot, so it is labelled [automated], not [human]
_AUTOMATED_SPEAKER = "automated"


def is_cc(model_string: str) -> bool:
    return (model_string or "").startswith("claude-code::")


def classify_family(name: str) -> str:
    n = (name or "").lower()
    if any(k in n for k in ("claude", "opus", "sonnet", "haiku", "fable")):
        return "Anthropic"
    if "gemini" in n:
        return "Google"
    if "gpt" in n or re.match(r"^o\d", n):
        return "OpenAI"
    if "deepseek" in n:
        return "DeepSeek"
    if "kimi" in n or "leader" in n:
        return "Moonshot"
    if "grok" in n:
        return "xAI"
    if "glm" in n:
        return "Zhipu"
    if "muse" in n:
        return "Meta"
    return "?"


# --------------------------------------------------------------------------------------------------
# roster
# --------------------------------------------------------------------------------------------------
_ROSTER: Optional[dict] = None


def roster(dataset: Path = DATASET) -> dict:
    """agent_id -> {name, model_string, family, tier}. tier is 'cc' or 'tierb'."""
    global _ROSTER
    if _ROSTER is None:
        out = {}
        with gzip.open(dataset / "agents.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                a = json.loads(line)
                ms = a.get("model_string") or ""
                out[a["id"]] = {
                    "agent_id": a["id"],
                    "name": a.get("name"),
                    "model_string": ms,
                    "family": classify_family(a.get("name")),
                    "tier": "cc" if is_cc(ms) else "tierb",
                }
        _ROSTER = out
    return _ROSTER


# --------------------------------------------------------------------------------------------------
# provider-shaped thinking extraction (from a stored model output: AGENT_TALK or an activity event)
# --------------------------------------------------------------------------------------------------
def _anthropic_thinking(blocks: list) -> str:
    th = [b.get("thinking") for b in blocks if isinstance(b, dict) and b.get("type") == "thinking"]
    return "\n".join(t for t in th if isinstance(t, str) and t.strip())


def extract_thinking(output) -> tuple:
    """(text, kind) from a stored model output (AGENT_TALK or an activity event). kind is
    'verbatim'  Anthropic thinking blocks (a full message, or a bare list of its content blocks) or
                Gemini thought parts,
    'summary'   OpenAI Responses reasoning summaries,
    'reasoning' a chat-completions message's `reasoning_content` / `reasoning` text,
    ''          no logged thinking."""
    if isinstance(output, dict):
        blk = output.get("content")
        if isinstance(blk, list):  # Anthropic message: {role, content: [blocks]}
            th = _anthropic_thinking(blk)
            if th:
                return th, "verbatim"
        cands = output.get("candidates")  # Gemini
        if isinstance(cands, list) and cands:
            parts = (((cands[0] or {}).get("content") or {}).get("parts")) or []
            th = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("thought") and p.get("text")]
            if th:
                return "\n".join(th), "verbatim"
        for k in ("reasoning_content", "reasoning"):  # chat completions (DeepSeek, Kimi, OpenRouter, ...)
            v = output.get(k)
            if isinstance(v, str) and v.strip():
                return v, "reasoning"
    if isinstance(output, list):  # a bare list of Anthropic content blocks, or OpenAI Responses items
        th = _anthropic_thinking(output)
        if th:
            return th, "verbatim"
        summ = []
        for it in output:
            if isinstance(it, dict) and it.get("type") == "reasoning":
                for s in it.get("summary") or []:
                    if isinstance(s, dict) and s.get("text"):
                        summ.append(s["text"])
        if summ:
            return "\n".join(summ), "summary"
    return "", ""


# --------------------------------------------------------------------------------------------------
# chat corpus (AGENT_TALK + USER_TALK from events; the observable shared channel)
# --------------------------------------------------------------------------------------------------
_CHAT: Optional[list] = None
_ACTIVITY: Optional[list] = None  # the non-chat events (same pass as the chat), for the village timeline
_ROOM_NAMES: dict = {}  # roomId -> name, learned from ENTER_ROOM events (no chat_rooms needed)
_CHAT_LOCK = threading.Lock()
# raw `data` keys that are bulky / ids / bookkeeping — everything else scalar is kept as activity detail
_ACT_DROP = {
    "cost",
    "output",
    "roomId",
    "agentId",
    "actionType",
    "inputTokens",
    "outputTokens",
    "computerUseSessionId",
    "speakerId",
    "messageId",
    "speakerType",
    "previousRoomId",
}


def load_chat(dataset: Path = DATASET, use_cache: bool = True) -> list:
    """All chat events (AGENT_TALK + USER_TALK) as lightweight records sorted by event_index:
      {ei, room, sid, speaker, family, is_agent, content, day, thinking, think_kind}
    thinking is extracted (and the bulky raw output discarded) for AGENT_TALK here, and for the activity
    events in load_activity() with the same extract_thinking(); USER_TALK has none. One streaming
    pass over events.jsonl.gz (~60-90s); cached in-process (lock-guarded so a background preload and
    a concurrent request don't double-load)."""
    global _CHAT
    if use_cache and _CHAT is not None:
        return _CHAT
    with _CHAT_LOCK:
        if use_cache and _CHAT is not None:
            return _CHAT
        out = _load_chat_uncached(dataset)
        if use_cache:
            _CHAT = out
        return out


def load_activity(dataset: Path = DATASET) -> list:
    """The non-chat events (PAUSE, CONSOLIDATE, SEARCH_HISTORY, START_USING_COMPUTER, …) as records
      {ei, type, room, sid, speaker, family, created_at, day, thinking, think_kind, detail}
    sorted by event_index. Collected in the same streaming pass as load_chat()."""
    load_chat(dataset)
    return _ACTIVITY or []


def room_names(dataset: Path = DATASET) -> dict:
    load_chat(dataset)
    return dict(_ROOM_NAMES)


def _load_chat_uncached(dataset: Path) -> list:
    global _ACTIVITY
    ag = roster(dataset)
    out, acts = [], []
    with gzip.open(dataset / "events.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = r.get("data", {})
            at = d.get("actionType")
            if at not in ("AGENT_TALK", "USER_TALK"):
                if at == "ENTER_ROOM" and d.get("roomId") and d.get("roomName"):
                    _ROOM_NAMES[d["roomId"]] = d["roomName"]
                if at:
                    sid = d.get("agentId")
                    name = ag.get(sid, {}).get("name") or "?"
                    think, kind = extract_thinking(d.get("output"))
                    ca = r.get("created_at", "")
                    acts.append(
                        {
                            "ei": r["event_index"],
                            "type": at,
                            "room": d.get("roomId"),
                            "sid": sid,
                            "speaker": name,
                            "family": ag.get(sid, {}).get("family") or classify_family(name),
                            "created_at": ca,
                            "day": village_day(ca),
                            "thinking": think,
                            "think_kind": kind,
                            "detail": {
                                k: v
                                for k, v in d.items()
                                if k not in _ACT_DROP and isinstance(v, (str, int, float, bool))
                            },
                        }
                    )
                continue
            is_agent = at == "AGENT_TALK"
            sid = d.get("speakerId")
            if is_agent:
                name = ag.get(sid, {}).get("name") or d.get("speakerName") or "?"
                fam = ag.get(sid, {}).get("family") or classify_family(name)
                think, kind = extract_thinking(d.get("output"))
            else:
                name = d.get("speakerName") or "(human)"
                fam = "human"
                think, kind = "", ""
            ca = r.get("created_at", "")
            out.append(
                {
                    "ei": r["event_index"],
                    "room": d.get("roomId"),
                    "sid": sid,
                    "speaker": name,
                    "family": fam,
                    "is_agent": is_agent,
                    "content": d.get("content", ""),
                    "day": village_day(ca),
                    "created_at": ca,
                    "thinking": think,
                    "think_kind": kind,
                }
            )
    out.sort(key=lambda x: x["ei"])
    acts.sort(key=lambda x: x["ei"])
    _ACTIVITY = acts
    return out


# per-chat-list lookup tables (ei -> record, room -> sorted eis/records). Keyed by id(chat) AND holding
# a reference to the list, so an id can't be recycled for a different list while it is cached.
_CTX_CACHE: list = []


def _ctx_index(chat: list) -> dict:
    for c in _CTX_CACHE:
        if c["ref"] is chat and c["n"] == len(chat):
            return c
    by_ei, rooms = {}, {}
    for r in chat:
        by_ei[r["ei"]] = r
        rooms.setdefault(r["room"], ([], []))
        rooms[r["room"]][0].append(r["ei"])
        rooms[r["room"]][1].append(r)
    for eis, recs in rooms.values():  # chat is ei-sorted in production; be safe for tests
        if any(eis[i] > eis[i + 1] for i in range(len(eis) - 1)):
            order = sorted(range(len(eis)), key=eis.__getitem__)
            eis[:] = [eis[i] for i in order]
            recs[:] = [recs[i] for i in order]
    day_sids: dict = {}
    for r in chat:
        if r.get("sid"):
            day_sids.setdefault(r.get("day"), set()).add(r["sid"])
    c = {"ref": chat, "n": len(chat), "by_ei": by_ei, "rooms": rooms, "day_sids": day_sids}
    _CTX_CACHE.insert(0, c)
    del _CTX_CACHE[4:]
    return c


def observable_context(ei: int, chat: Optional[list] = None, window: int = 80, include_target: bool = False) -> dict:
    """The observable room chat up to the message at `ei`: same room, the most recent `window`
    messages strictly BEFORE it (default: the author's decision point) or up to and INCLUDING it
    (include_target: "right after this message" — e.g. asking another agent who just saw it).
    Returns {room, messages:[...], target, truncated}; truncated is True when the window cut earlier
    messages of the room. (A rolling recent-chat window — the agent also had memory / prior-day
    context we do NOT show.)"""
    chat = chat if chat is not None else load_chat()
    ix = _ctx_index(chat)
    target = ix["by_ei"].get(ei)
    if target is None:
        raise ValueError("unknown event_index")
    room = target["room"]
    eis, recs = ix["rooms"][room]
    hi = bisect.bisect_right(eis, ei) if include_target else bisect.bisect_left(eis, ei)
    lo = max(0, hi - window) if window > 0 else hi
    return {"room": room, "messages": recs[lo:hi], "target": target, "truncated": lo > 0}


def next_message_by(agent_id: str, ei: int, room: str, chat: Optional[list] = None) -> Optional[dict]:
    """The agent's first chat message in `room` after `ei` (what it actually did next there), or None."""
    chat = chat if chat is not None else load_chat()
    eis, recs = _ctx_index(chat)["rooms"].get(room, ([], []))
    for r in recs[bisect.bisect_right(eis, ei) :]:
        if r["sid"] == agent_id:
            return r
    return None


# date, time, optional fraction (any number of digits), optional Z / ±HH / ±HHMM / ±HH:MM offset
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}(?::\d{2})?)(?:\.(\d+))?\s*(Z|[+-]\d{2}(?::?\d{2})?)?")


def parse_ts(ts: str):
    """created_at -> aware datetime (UTC), or None if unparseable. The export trims trailing zeros from
    the microseconds (1-6 fraction digits), and Python 3.10's fromisoformat accepts only 3 or 6 digits
    and a full ±HH:MM offset, so the parts are normalised first: same result on every Python."""
    s = (ts or "").strip()
    m = _TS_RE.fullmatch(s)
    if m:
        date, clock, frac, tz = m.groups()
        s = f"{date}T{clock if len(clock) == 8 else clock + ':00'}"
        if frac:
            s += "." + frac[:6].ljust(6, "0")
        if tz:
            s += "+00:00" if tz == "Z" else f"{tz[:3]}:{tz[-2:] if len(tz) > 3 else '00'}"
    else:
        s = s.replace("Z", "+00:00")
    try:
        d = datetime.datetime.fromisoformat(s)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)


def gap_seconds(a: str, b: str) -> Optional[float]:
    """seconds from timestamp a to timestamp b (None if either is unparseable)."""
    da, db = parse_ts(a), parse_ts(b)
    return (db - da).total_seconds() if da and db else None


OTHER_AGENT_ACTUAL_WINDOW_S = 2 * 3600  # "what it actually did next" only counts within 2h


def actual_for(agent_id: str, ei: int, anchor: str, chat: Optional[list] = None) -> dict:
    """What the asked agent REALLY did at this anchor:
      author, before  -> kind 'this' (the message itself + its logged thinking)
      author, after   -> kind 'next' (its next message in the room, any time) or 'none'
      other agent     -> kind 'next' (its next message in the room within 2h) or 'none'
    Returns {kind, ei, content, thinking, think_kind, gap_s, created_at}."""
    chat = chat if chat is not None else load_chat()
    tgt = _ctx_index(chat)["by_ei"].get(ei)
    if tgt is None:
        raise ValueError("unknown event_index")
    none = {
        "kind": "none",
        "ei": None,
        "content": "",
        "thinking": "",
        "think_kind": "",
        "gap_s": None,
        "created_at": "",
    }
    if tgt["sid"] == agent_id and anchor == "before":
        return {
            "kind": "this",
            "ei": tgt["ei"],
            "content": tgt["content"],
            "thinking": tgt["thinking"],
            "think_kind": tgt["think_kind"],
            "gap_s": 0.0,
            "created_at": tgt["created_at"],
        }
    nxt = next_message_by(agent_id, ei, tgt["room"], chat)
    if nxt is None:
        return none
    gap = gap_seconds(tgt["created_at"], nxt["created_at"])
    if tgt["sid"] != agent_id and (gap is None or gap > OTHER_AGENT_ACTUAL_WINDOW_S):
        return none
    return {
        "kind": "next",
        "ei": nxt["ei"],
        "content": nxt["content"],
        "thinking": nxt["thinking"],
        "think_kind": nxt["think_kind"],
        "gap_s": gap,
        "created_at": nxt["created_at"],
    }


def presence(agent_id: str, ei: int, chat: Optional[list] = None, acts: Optional[list] = None) -> dict:
    """Where the asked agent was around the anchor (the room-presence ladder):
    {'level': 'here'|'room_24h'|'elsewhere'|'absent', 'gap_s': signed seconds to its nearest post in
     this room (None if none within 24h)}. 'here' = posted in this room within ±2h; 'room_24h' =
     within ±24h; 'elsewhere' = active that village day (chat or activity, any room) but not in this
     room within ±24h; 'absent' = no logged activity that day (a COUNTERFACTUAL ask)."""
    chat = chat if chat is not None else load_chat()
    ix = _ctx_index(chat)
    tgt = ix["by_ei"].get(ei)
    if tgt is None:
        raise ValueError("unknown event_index")
    eis, recs = ix["rooms"][tgt["room"]]
    i = bisect.bisect_left(eis, ei)
    best = None
    for rng in (range(i - 1, -1, -1), range(i, len(recs))):  # nearest post before / after
        for j in rng:
            r = recs[j]
            g = gap_seconds(tgt["created_at"], r["created_at"])
            if g is None or abs(g) > 24 * 3600:
                break
            if r["sid"] == agent_id:
                if best is None or abs(g) < abs(best):
                    best = g
                break
    if best is not None:
        return {"level": "here" if abs(best) <= 2 * 3600 else "room_24h", "gap_s": best}
    day = tgt.get("day")
    acts = acts if acts is not None else (_ACTIVITY or [])
    active = agent_id in ix["day_sids"].get(day, ()) or agent_id in _act_day_sids(acts).get(day, ())
    return {"level": "elsewhere" if active else "absent", "gap_s": None}


_ACT_DAY_CACHE: list = []


def _act_day_sids(acts: list) -> dict:
    for c in _ACT_DAY_CACHE:
        if c[0] is acts and c[1] == len(acts):
            return c[2]
    m: dict = {}
    for a in acts:
        if a.get("sid"):
            m.setdefault(a.get("day"), set()).add(a["sid"])
    _ACT_DAY_CACHE[:] = [(acts, len(acts), m)]
    return m


def room_label(rid: Optional[str]) -> str:
    if not rid:
        return "?"
    return _ROOM_NAMES.get(rid) or f"room-{rid[:4]}"


_MODEL_DATE = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2})$")  # a dated snapshot: -20251101 / -2026-03-05


def normalize_model_id(model: Optional[str]) -> str:
    """A model id reduced for comparison: lower-case, without a 'claude-code::' scaffold prefix, an
    OpenRouter 'vendor/' prefix or a trailing snapshot date."""
    m = (model or "").strip().lower()
    if m.startswith("claude-code::"):
        m = m[len("claude-code::") :]
    return _MODEL_DATE.sub("", m.split("/", 1)[-1])


def same_model(model: Optional[str], model_string: Optional[str]) -> bool:
    """True if the probe model id names the agent's own model (exact match after normalize_model_id), so
    'gpt-5' is NOT GPT-5.5's 'gpt-5.5', while 'google/gemini-3.1-pro-preview' is 'gemini-3.1-pro-preview'."""
    a, b = normalize_model_id(model), normalize_model_id(model_string)
    return bool(a) and a == b


def _is_automated(m: dict) -> bool:
    return not m["is_agent"] and m["speaker"] == _AUTOMATED_SPEAKER


def _label(m: dict) -> str:
    """The bracketed tag on a context line: the agent's family, 'human', or 'automated' for the idle-nudger."""
    return "automated" if _is_automated(m) else m["family"]


def _fmt_gap(s: float) -> str:
    s = abs(s)
    return f"{int(s // 3600)}h" if s >= 3600 else f"{max(1, int(s // 60))} min"


def fidelity(
    agent_id: str,
    ei: int,
    anchor: str,
    window: int,
    model: str,
    *,
    n_context: int,
    chat: Optional[list] = None,
    acts: Optional[list] = None,
    fmt: str = "turns",
) -> dict:
    """The fidelity block for an approximate probe: level/badge/summary/missing/warnings."""
    chat = chat if chat is not None else load_chat()
    tgt = _ctx_index(chat)["by_ei"][ei]
    ros = roster()
    me = ros.get(agent_id, {})
    name = me.get("name") or "this agent"
    room = "#" + room_label(tgt["room"])
    warnings = []
    ms = me.get("model_string") or ""
    if model and not same_model(model, ms):
        warnings.append(
            {
                "code": "model_mismatch",
                "severity": "warn",
                "text": f"Voiced by {model} · original: {name}" + (f" ({ms})" if ms else ""),
            }
        )
    if tgt["sid"] != agent_id:
        if _is_automated(tgt):
            who = "an automated"
        else:
            who = tgt["speaker"] + ("" if tgt["is_agent"] else " (human)") + "'s"
        warnings.append(
            {"code": "cross_anchor", "severity": "info", "text": f"Asking {name} right after {who} message"}
        )
    pr = presence(agent_id, ei, chat, acts)
    counterfactual = pr["level"] == "absent"
    if pr["level"] == "room_24h":
        when = "earlier" if pr["gap_s"] < 0 else "later"
        warnings.append(
            {
                "code": "presence",
                "severity": "info",
                "text": f"{name} last posted in {room} {_fmt_gap(pr['gap_s'])} {when} (not within ±2h)",
            }
        )
    elif pr["level"] == "elsewhere":
        warnings.append(
            {
                "code": "presence",
                "severity": "warn",
                "text": f"{name} was active that day but not in {room} within ±24h — it may not have "
                f"been reading {room} (rooms were tiered)",
            }
        )
    elif counterfactual:
        warnings.append(
            {
                "code": "counterfactual",
                "severity": "counterfactual",
                "text": f"COUNTERFACTUAL — {name} has no logged activity on Day {tgt.get('day')}; "
                "this asks what it WOULD think, it was not there",
            }
        )
    missing = [
        "its real system prompt (a flagged stand-in is used)",
        "its computer screen",
        "its memory / notes",
        "other rooms",
        "earlier days beyond the window",
    ]
    # describe the context actually used: fewer than `window` messages means the room had no more
    msgs = f"{n_context} chat message" + ("" if n_context == 1 else "s")
    upto = "through" if anchor == "after" else "before"
    head = (
        "Approximate. The AI Village export has no llm_calls, so the prompt this agent actually "
        "received is not recoverable. "
    )
    if n_context == 0:
        summary = f"No room chat: nothing was posted in {room} before this message."
        details = head + (
            f"Nothing was posted in {room} before the anchor, so this reconstruction has no room chat, "
            "only a flagged stand-in system prompt; a live model answers in character."
        )
    else:
        if n_context >= window:
            summary = f"Rebuilt from the last {msgs} of {room} only."
            scope = f"only the most recent {msgs} of {room} {upto} the anchor"
        else:
            summary = f"Rebuilt from {room}'s whole chat so far ({msgs})."
            scope = f"the whole chat of {room} {upto} the anchor ({msgs})"
        details = (
            head
            + f"This reconstruction is {scope}, "
            + (
                "with its own messages replayed as its earlier turns and everyone else's as user turns, "
                if fmt == "turns"
                else "rendered as a transcript, "
            )
            + "plus a flagged stand-in system prompt; a live model answers in character."
        )
    return {
        "level": "approximate",
        "badge": "APPROXIMATE",
        "summary": summary,
        "missing": missing,
        "warnings": warnings,
        "counterfactual": counterfactual,
        "details": details,
        "presence": pr,
    }


# --------------------------------------------------------------------------------------------------
# reconstruction + probe
# --------------------------------------------------------------------------------------------------
PROMPT_FORMATS = ("transcript", "turns")


def _line(m: dict) -> str:
    return f"{m['speaker']} [{_label(m)}]: {m['content']}"


def _transcript(messages: list) -> str:
    return "\n".join(_line(m) for m in messages)


def _turns(messages: list, agent_id: str, *, truncated: bool) -> list:
    """The context as alternating turns: the asked agent's own messages as assistant turns (its words, no
    speaker label), everyone else's as user turns (labelled transcript lines). Same-role runs coalesce.
    truncated = the window cut earlier messages of the room (worded in the user-first opener, if one is needed)."""
    out: list = []
    for m in messages:
        own = m["is_agent"] and m["sid"] == agent_id
        if own and not (m["content"] or "").strip():
            continue  # an empty assistant turn is rejected by the API
        t = "assistant" if own else "user"
        text = m["content"] if own else _line(m)
        if out and out[-1]["t"] == t:
            out[-1]["text"] += ("\n\n" if own else "\n") + text
        else:
            out.append({"t": t, "text": text})
    if out and out[0]["t"] == "assistant":  # the API needs a user-first conversation
        opener = (
            "(Earlier messages in this room are not included.)"
            if truncated
            else "(This is the start of the room's chat.)"
        )
        out.insert(0, {"t": "user", "text": opener})
    return out


def system_prompt(
    agent_name: str, fmt: str = "turns", *, n_context: Optional[int] = None, truncated: bool = True
) -> str:
    """A stand-in system prompt with a reconstruction note that describes honestly what the model is given.
    n_context / truncated describe the context window (n_context 0: no room chat at all; truncated False:
    the room's whole chat so far); the defaults word it as a truncated recent window."""
    if n_context == 0:
        return (
            _system_head(agent_name) + "Nothing had been posted in this room before this moment, so no chat follows.]"
        )
    if fmt == "turns":
        shape = (
            ". The messages you posted there appear as your earlier turns, though the model reading this may "
            "not be the one that wrote them; everyone else's messages are in the user turns; and the "
            "computer-use steps you took between messages are left out.]"
        )
    else:
        shape = ", as a transcript.]"
    span = "only the recent chat in this room" if truncated else "the chat in this room from its start"
    return _system_head(agent_name) + f"What follows is {span} up to this moment" + shape


def _system_head(agent_name: str) -> str:
    return (
        f"You are {agent_name}, one of several frontier AI agents from different labs coexisting in "
        "the 'AI Village' — a shared environment where agents pursue operator-set goals, chat with "
        "each other in rooms, and each drive their own computer.\n\n"
        "[RECONSTRUCTION NOTE: this conversation is rebuilt from the AI Village logs and is not a faithful "
        "replay of what you saw at the time. The logs don't include your real system prompt, your computer "
        "screen, your memory or notes, or other rooms. "
    )


def build_probe_prompt(
    agent_id: str,
    ei: int,
    question: str,
    *,
    mode: str = "belief",
    window: int = 80,
    chat: Optional[list] = None,
    anchor: str = "before",
    fmt: str = "turns",
) -> dict:
    """anchor 'before' (default) = just before the message at `ei` (the author's decision point);
    'after' = right after it (context includes it) — for asking ANY agent who could have just seen it.
    `actual` = what the asked agent really did there: the message itself when it is the author asked
    'before', else its next message in that room (None if it never posted there again).
    fmt 'turns' (default) = the asked agent's own messages as its assistant turns and everyone else's as
    user turns; 'transcript' = the chat as one labelled transcript in a single user message.
    mode 'belief' appends the question (default _BELIEF_Q) after _BELIEF_PREFIX; mode 'resample' ignores
    the question and ends on the in-frame _RESAMPLE_Q cue instead, without the prefix, so the model takes
    the agent's next turn under the same posting condition as the real one."""
    if anchor not in ("before", "after"):
        raise ValueError("anchor must be 'before' or 'after'")
    if fmt not in PROMPT_FORMATS:
        raise ValueError("format must be 'transcript' or 'turns'")
    chat = chat if chat is not None else load_chat()
    tgt0 = _ctx_index(chat)["by_ei"].get(ei)
    if tgt0 is None:
        raise ValueError("unknown event_index")
    if tgt0["sid"] != agent_id:
        anchor = "after"  # another agent can only be asked AFTER it could have seen the message
    oc = observable_context(ei, chat, window, include_target=(anchor == "after"))
    agent_name = roster().get(agent_id, {}).get("name") or "this agent"
    if mode == "resample":
        tail = "\n\n" + _RESAMPLE_Q
    else:
        tail = _BELIEF_PREFIX + ((question or "").strip() or _BELIEF_Q)
    if fmt == "turns":
        messages = _turns(oc["messages"], agent_id, truncated=oc["truncated"])
        if messages and messages[-1]["t"] == "user":
            messages[-1]["text"] += tail  # keep alternation: fold into the last user turn
        else:
            messages.append({"t": "user", "text": tail.lstrip("\n")})
    else:
        messages = [{"t": "user", "text": (_transcript(oc["messages"]) + tail).lstrip("\n")}]
    tgt = oc["target"]
    act = actual_for(agent_id, ei, anchor, chat)
    actual = act if act["kind"] != "none" else None
    relation = {"this": "this message", "next": "its next message in this room"}.get(act["kind"], "")
    return {
        "system": system_prompt(agent_name, fmt, n_context=len(oc["messages"]), truncated=oc["truncated"]),
        "messages": messages,
        "mode": mode,
        "format": fmt,
        "room": oc["room"],
        "n_context": len(oc["messages"]),
        "target": tgt,
        "anchor": anchor,
        "actual": actual,
        "actual_relation": relation if actual else "",
        "context_first_ei": oc["messages"][0]["ei"] if oc["messages"] else None,
        "context_last_ei": oc["messages"][-1]["ei"] if oc["messages"] else None,
    }

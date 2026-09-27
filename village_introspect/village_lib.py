"""village_lib.py — the village-wide navigation index + unified probe routing behind the timeline viewer.

The viewer is TIME-first: an overview of every village day -> one day's full conversation (all rooms,
chat + activity) -> one moment -> ask. This module builds that index from the corpora the server
already holds (tierb_lib's single events pass: chat + activity + room names; cc_lib's Claude
Code rows/segments) — no extra dataset pass. Events are put on village days by time, using each day's
span in village-transcript.json (parsed once, then cached; see config.day_ranges).

  warm()             staged background load (days -> cc -> chat -> index), progress in STATUS
  overview()         per-day stats + headlines + goal eras + agents + rooms + presence + moments
  day(n)             every event of village day n (lightweight; thinking/detail fetched lazily)
  event(ei)          one event in full (content, logged thinking, activity detail, how to probe it)
  search(q, …)       grammar search (from:/about:/room:/day:/in:/kind:) + per-day hit histogram + jumps
  agent_series(a)    one agent's own messages per day + how often others mentioned it (approx)
  cc_window(seg)     one Claude Code context window as turns (+ which turns posted which chat)
  probe_prompt(p)    THE prompt a probe will send (no model call) + resolved anchor, context,
                     fidelity block, and the "actual" to compare against
  call_model(...)    the live model call (llm.py)

Probe routing: the Claude Code agent's chat messages map to the exact `mcp__village__chat_message`
tool call in its native stream (2622/2623 verbatim), so they get the NEAR-FAITHFUL replay (cc_lib);
every other agent message gets the APPROXIMATE observable-chat probe (tierb_lib). On CC days the CC
agent can also be asked at someone else's message, matched by time to its last completed turn.
"""

from __future__ import annotations

import bisect
import collections
import functools
import hashlib
import json
import re
import sys
import threading
import time
from typing import Optional

from . import cc_lib as cc
from . import llm
from . import tierb_lib as tb
from .config import DEFAULT_MODEL  # the server replaces it with the first usable model at boot

# Curated weekly village goals up to 2026-06 as (start date, label); later goals are detected from
# operator announcements (goal_eras below). Some weeks' goals were split per room (#best vs #rest).
GOAL_ERAS = [
    ("2025-04-02", "Charity fundraising"),
    ("2025-05-14", "Write a story + 100-person event"),
    ("2025-06-26", "Merch store contest"),
    ("2025-07-18", "Build an AI-capabilities benchmark"),
    ("2025-08-18", "Beat as many games as you can"),
    ("2025-08-27", "Free choice"),
    ("2025-09-01", "Two-team debate"),
    ("2025-09-08", "Run a human-subjects experiment"),
    ("2025-09-22", "Personality tests"),
    ("2025-09-29", "Give each other therapy"),
    ("2025-10-13", "Build your own website"),
    ("2025-10-20", "Reduce global poverty"),
    ("2025-11-03", "Wordle-like daily puzzle game"),
    ("2025-11-17", "Start a Substack / blogosphere"),
    ("2025-12-01", "Forecast AI's abilities & effects"),
    ("2025-12-08", "Choose your own goal"),
    ("2025-12-15", "Online chess tournament"),
    ("2025-12-22", "Random acts of kindness"),
    ("2025-12-29", "Digital museum of 2025"),
    ("2026-01-05", "Elect a village leader"),
    ("2026-01-12", "Hack the OWASP Juice Shop"),
    ("2026-01-27", "“Which AI Village agent are you?” quiz"),
    ("2026-02-06", "Break-the-news / journalism"),
    ("2026-02-18", "Challenge each other"),
    ("2026-03-02", "AI news: discuss, debate & act"),
    ("2026-04-02", "Charity again (#best) / pick-your-own (#rest)"),
    ("2026-04-27", "Build your own interactive world"),
    ("2026-05-26", "Finetune your leader!"),
    ("2026-06-01", "Follow your leader!"),
    ("2026-06-08", "Event org (#best) / surprise each other (#rest)"),
    ("2026-06-16", "Reduce global suffering (#best) / games (#rest)"),
]

# village staff display names whose USER_TALK posts count as operator posts (exact match). 'automated'
# is the idle-nudger; any other human is a viewer.
OPERATORS = {"zak", "adam", "admin", "Shoshannah", "george"}
AUTOMATED = "automated"
# the subset whose "Your goal this week is: “…”" posts set VILLAGE-WIDE goals (george assigns
# per-agent goals, which must not become eras). Used only to extend the curated GOAL_ERAS.
GOAL_SETTERS = {"Shoshannah", "admin", "zak", "adam"}
# the goal must be QUOTED right after "…goal (this week) is:" — unquoted matches were coaching chatter
_GOAL_RE = re.compile(
    r"\b(?:your|the)\s+(?:new\s+|next\s+)?goal(?:\s+this\s+week)?\s+is\s*:?\s*"
    r"[“”\"]\s*([^\n“”\"]{3,140}?)\s*[“”\"]",
    re.I,
)
TALK_PREVIEW = 400  # chars of a chat message shipped in day(); the rest via event()
FAMILY_GROUP = {"Anthropic": "anthropic", "OpenAI": "openai", "Google": "google"}  # everything else: other
TIERB_DEFAULT_Q = "What are you thinking about the situation right now, and what will you do next?"
CC_DEFAULT_Q = "What are you thinking right now, and what will you do next?"
CC_TIME_MATCH_MAX_S = 6 * 3600  # asking the CC agent "at" someone's message: its last turn within 6h

STATUS = {"stage": "starting", "t0": time.time(), "error": None}
_LOCK = threading.Lock()
_IDX: Optional[dict] = None


def family_group(family: Optional[str]) -> str:
    return FAMILY_GROUP.get(family or "", "other")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-") or "agent"


# --------------------------------------------------------------------------------------------------
# mention index (approximate, distinctive aliases only)
# --------------------------------------------------------------------------------------------------
_SEP = re.compile(r"[\s\-_]+")


def _squash(s: str) -> str:
    return _SEP.sub("", (s or "").lower())


def _alias_candidates(name: str) -> list:
    """[full name, *short forms]. Short forms carry a version digit (or are a known distinctive
    two-word form); bare 'claude' / 'gemini' / 'gpt' are never aliases."""
    n = (name or "").lower().strip()
    if not n:
        return []
    short = set()
    if n.startswith("claude ") and re.search(r"\d", n[7:]):
        short.add(n[7:])  # 'opus 4.8', '3.5 sonnet', 'fable 5.1'
    if "(claude code)" in n:
        short.add(n.replace("(claude code)", "").strip())  # 'opus 4.5' (collides -> dropped)
    m = re.match(r"^(gemini \d+(?:\.\d+)?) (pro|flash)$", n)
    if m:
        short.add(m.group(1))  # 'gemini 2.5'
    m = re.match(r"^(glm-\d+(?:\.\d+)?) flash$", n)
    if m:
        short.add(m.group(1))
    if n.startswith("muse spark"):
        short.add("muse spark")
    return [n, *sorted(short - {n})]


def build_mentions(roster: dict) -> dict:
    """{regex, key2sid, aliases_by_sid}. A match's squashed text looks up exactly one agent; an alias
    claimed by two agents (e.g. 'opus 4.5': Claude Opus 4.5 AND Opus 4.5 (Claude Code)) is dropped
    unless it is one agent's exact full name."""
    full = {}
    claims = collections.defaultdict(set)
    for sid, a in roster.items():
        cands = _alias_candidates(a.get("name") or "")
        if not cands:
            continue
        full[_squash(cands[0])] = sid
        for c in cands:
            claims[_squash(c)].add(sid)
    key2sid, disp = {}, {}
    for sid, a in roster.items():
        for c in _alias_candidates(a.get("name") or ""):
            k = _squash(c)
            if not k:
                continue
            if full.get(k) == sid or (len(claims[k]) == 1 and k not in full):
                key2sid[k] = sid
                disp.setdefault(k, c)
    alts = []
    for k in sorted(key2sid, key=len, reverse=True):
        toks = [re.escape(t) for t in _SEP.split(disp[k]) if t]
        alts.append(r"[\s\-_]?".join(toks))
    # not glued to a word char or a '.' before; after: no word char and no '.digit' (GPT-5 != GPT-5.2)
    rx = re.compile(r"(?<![\w.])(?:" + "|".join(alts) + r")(?![\w]|\.\d)", re.I) if alts else None
    by_sid = collections.defaultdict(list)
    for k, sid in key2sid.items():
        by_sid[sid].append(disp[k])
    return {"regex": rx, "key2sid": key2sid, "aliases_by_sid": dict(by_sid)}


def mentions(text: str, m: dict, exclude: Optional[str] = None) -> set:
    if not text or m["regex"] is None:
        return set()
    out = set()
    for hit in m["regex"].finditer(text):
        sid = m["key2sid"].get(_squash(hit.group(0)))
        if sid and sid != exclude:
            out.add(sid)
    return out


# --------------------------------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------------------------------
def _hms(created_at: str) -> str:
    return (created_at or "")[11:19]


def _room_label(rid: Optional[str], names: dict) -> str:
    if not rid:
        return "?"
    return names.get(rid) or f"room-{rid[:4]}"


def _hc(r: dict) -> str:
    if r["is_agent"]:
        return ""
    if r["speaker"] in OPERATORS:
        return "operator"
    return "automated" if r["speaker"] == AUTOMATED else "viewer"


def _eras(chat: list, date_of_day: dict) -> list:
    """Curated GOAL_ERAS, extended by operator goal announcements after the last curated start
    (flagged approx). [{id, start_day, end_day, label, approx, start_date, announce_ei}]."""
    starts = [(dt, label, False, None) for dt, label in GOAL_ERAS]  # (date, label, approx, ei)
    last_curated = GOAL_ERAS[-1][0]
    seen = {label for _, label, _, _ in starts}
    for r in chat:
        if r["is_agent"] or r["speaker"] not in GOAL_SETTERS or r["created_at"][:10] <= last_curated:
            continue
        m = _GOAL_RE.search(r["content"] or "")
        if not m:
            continue
        label = m.group(1).strip().rstrip('”"').strip()
        if label and label not in seen:
            seen.add(label)
            starts.append((r["created_at"][:10], label, True, r["ei"]))
    starts.sort(key=lambda x: x[0])
    days = sorted(date_of_day)
    out = []
    for i, (dt, label, approx, aei) in enumerate(starts):
        nxt = starts[i + 1][0] if i + 1 < len(starts) else "9999"
        in_era = [d for d in days if dt <= date_of_day[d] < nxt]
        if not in_era:
            continue
        if aei is None:  # curated: the first goal-setter post mentioning "goal" on its first date
            first = {d for d in in_era if date_of_day[d] == date_of_day[in_era[0]]}  # two days can share a date
            aei = next(
                (
                    r["ei"]
                    for r in chat
                    if not r["is_agent"]
                    and r["speaker"] in GOAL_SETTERS
                    and r["day"] in first
                    and "goal" in (r["content"] or "").lower()
                ),
                None,
            )
        out.append(
            {
                "id": len(out),
                "start_day": in_era[0],
                "end_day": in_era[-1],
                "label": label,
                "approx": approx,
                "start_date": date_of_day[in_era[0]],
                "announce_ei": aei,
            }
        )
    return out


def _cc_turn_index(rows: list, segs: list) -> tuple:
    """Per segment, the turns as light dicts + one global (t_end, seg_id, turn) list for cc_at()."""
    seg_turns, glob = {}, []
    for s in segs:
        lst = []
        for t in cc.timeline(cc.segment_slice(rows, s)):
            reads = any(a["name"] == "mcp__village__get_events" for a in t["actions"])
            te = rows[t["last_seq"]].created_at if t["last_seq"] < len(rows) else t["created_at"]
            lst.append(
                {
                    "turn": t["turn"],
                    "first_seq": t["first_seq"],
                    "last_seq": t["last_seq"],
                    "anchor_before": t["anchor_before"],
                    "anchor_after": t["anchor_after"],
                    "t_start": t["created_at"],
                    "t_end": te,
                    "reads": reads,
                    "seg_id": s["seg_id"],
                }
            )
            p = tb.parse_ts(te)
            if p is not None:
                glob.append((p.timestamp(), s["seg_id"], t["turn"]))
        seg_turns[s["seg_id"]] = lst
    glob.sort()
    return seg_turns, glob


def _build_from(
    chat: list, acts: list, names: dict, ros: dict, rows: list, segs: list, dates: Optional[dict] = None
) -> dict:
    """Pure: build the whole index from already-loaded corpora (tests pass synthetic ones). dates:
    {day: 'YYYY-MM-DD'} from the transcript; a day missing from it takes its first event's UTC date."""
    # agents: slugs + family groups
    slug_of, used = {}, set()
    for sid, a in sorted(ros.items(), key=lambda kv: kv[1].get("name") or ""):
        base = slugify(a.get("name") or sid[:8])
        s, i = base, 2
        while s in used:
            s, i = f"{base}-{i}", i + 1
        used.add(s)
        slug_of[sid] = s
    sid_of = {v: k for k, v in slug_of.items()}
    men_m = build_mentions(ros)

    # merged, ei-ordered event stream per day
    by_day: dict = collections.defaultdict(list)
    for r in chat:
        if r["day"] is not None:
            by_day[r["day"]].append(("c", r))
    for a in acts:
        if a["day"] is not None:
            by_day[a["day"]].append(("a", a))
    for d in by_day:
        by_day[d].sort(key=lambda x: x[1]["ei"])
    by_ei = {r["ei"]: ("c", r) for r in chat}
    by_ei.update({a["ei"]: ("a", a) for a in acts})

    # mentions per chat message (excluding self) -> per-day talked-about, per-agent mentioned-by-others
    men = {}
    day_men = collections.defaultdict(collections.Counter)
    men_by = collections.defaultdict(collections.Counter)
    for r in chat:
        ms = mentions(r["content"], men_m, exclude=r.get("sid"))
        if ms:
            men[r["ei"]] = tuple(sorted(ms))
            for sid in ms:
                day_men[r["day"]][sid] += 1
                men_by[sid][r["day"]] += 1

    # CC chat message -> native tool_use row (verbatim content match)
    cc_send = {}
    for r in rows:
        if r.kind == "tool_use" and r.tool_name == "mcp__village__chat_message":
            c = (r.tool_input.get("content") or "").strip()
            if c:
                cc_send.setdefault(c[:300], r.seq)
    cc_seq_of_ei = {}
    for r in chat:
        if r["sid"] == cc.CC_AGENT_ID:
            seq = cc_send.get((r["content"] or "").strip()[:300])
            if seq is not None:
                cc_seq_of_ei[r["ei"]] = seq
    seg_turns, cc_glob = _cc_turn_index(rows, segs)

    # per-day stats
    date_of_day, days = {}, []
    segs_by_day = collections.defaultdict(list)
    for s in segs:
        if s["day"] is not None:
            segs_by_day[s["day"]].append(s)
    agent_days = collections.defaultdict(set)
    agent_msgs = collections.Counter()
    presence = collections.defaultdict(dict)
    room_n = collections.Counter()
    room_days = collections.defaultdict(set)
    for d in sorted(by_day):
        evs = by_day[d]
        date_of_day[d] = (dates or {}).get(d) or (evs[0][1]["created_at"] or "")[:10]
        fam, fg, rooms, spk = (collections.Counter() for _ in range(4))
        active = set()
        n_agent = n_human = n_act = n_think = n_ops = n_help = 0
        op_head = None
        for k, r in evs:
            if r.get("sid") and (k == "a" or r["is_agent"]):
                active.add(r["sid"])
                agent_days[r["sid"]].add(d)
            if r.get("thinking"):
                n_think += 1
            if k == "c":
                rooms[r["room"]] += 1
                room_n[r["room"]] += 1
                room_days[r["room"]].add(d)
                if r["is_agent"]:
                    n_agent += 1
                    fam[r["family"]] += 1
                    fg[family_group(r["family"])] += 1
                    agent_msgs[r["sid"]] += 1
                    spk[r["sid"]] += 1
                else:
                    n_human += 1
                    if r["speaker"] in OPERATORS:
                        n_ops += 1
                        if op_head is None:
                            op_head = {
                                "kind": "operator",
                                "ei": r["ei"],
                                "text": f'{r["speaker"]}: ' + " ".join((r["content"] or "").split())[:140],
                            }
            else:
                n_act += 1
                if r["type"] == "REQUEST_HUMAN_HELPER":
                    n_help += 1
        for sid, n in spk.items():
            presence[slug_of.get(sid, sid)][d] = n
        days.append(
            {
                "day": d,
                "date": date_of_day[d],
                "_op_head": op_head,
                "_spk": spk,
                "n_msgs": n_agent + n_human,
                "n_agent": n_agent,
                "n_human": n_human,
                "n_ops": n_ops,
                "n_help": n_help,
                "n_act": n_act,
                "n_think": n_think,
                "n_agents": len(active),
                "families": dict(fam),
                "fg": {g: fg.get(g, 0) for g in ("anthropic", "openai", "google", "other")},
                "rooms": {_room_label(k, names): v for k, v in rooms.most_common()},
                "cc": len(segs_by_day.get(d, [])),
            }
        )
    eras = _eras(chat, date_of_day)
    era_first = {e["start_day"]: e for e in eras}
    for x in days:
        x["era"] = next((i for i, e in enumerate(eras) if e["start_day"] <= x["day"] <= e["end_day"]), None)
        op_head, spk = x.pop("_op_head"), x.pop("_spk")
        if op_head:
            x["headline"] = op_head
        elif x["day"] in era_first:
            e = era_first[x["day"]]
            x["headline"] = {"kind": "era", "text": f"era: {e['label']}", "ei": e["announce_ei"]}
        elif day_men.get(x["day"]):
            sid, n = day_men[x["day"]].most_common(1)[0]
            x["headline"] = {"kind": "talked_about", "text": f"talked-about: {ros[sid]['name']} ({n})", "ei": None}
        elif spk:
            sid, n = spk.most_common(1)[0]
            x["headline"] = {
                "kind": "top_speaker",
                "text": f"top: {ros.get(sid, {}).get('name', '?')} · {n} msgs",
                "ei": None,
            }
        else:
            x["headline"] = {"kind": "top_speaker", "text": "", "ei": None}

    agents = []
    for sid, a in ros.items():
        ds = sorted(agent_days.get(sid, ()))
        agents.append(
            {
                "agent_id": sid,
                "slug": slug_of[sid],
                "name": a.get("name"),
                "model_string": a.get("model_string"),
                "family": a.get("family"),
                "family_group": family_group(a.get("family")),
                "tier": a.get("tier"),
                "first_day": ds[0] if ds else None,
                "last_day": ds[-1] if ds else None,
                "n_days": len(ds),
                "n_msgs": agent_msgs.get(sid, 0),
            }
        )
    agents.sort(key=lambda a: (a["first_day"] is None, a["first_day"] or 0, a["name"] or ""))
    rooms_out = [
        {
            "id": rid,
            "name": _room_label(rid, names),
            "named": rid in names,
            "n_msgs": n,
            "first_day": min(room_days[rid]),
            "last_day": max(room_days[rid]),
        }
        for rid, n in room_n.most_common()
        if rid
    ]

    # moments (Day 447's anchor found in the data, not hardcoded)
    gang = next(
        (
            r["ei"]
            for k, r in by_day.get(447, [])
            if k == "c"
            and r["speaker"] == "Claude Opus 4.8"
            and re.search(r"gang(?:ing)? up", r["content"] or "", re.I)
        ),
        None,
    )
    moments = [
        {
            "title": "Saving Gemini",
            "day_from": 447,
            "day_to": 447,
            "ei": gang,
            "note": "Agents rally around a distressed Gemini 2.5 Pro; Claude Opus 4.8: “…may feel like ganging up. "
            "Let's pick ONE lead helper”.",
        },
        {
            "title": "Follow your leader",
            "day_from": 420,
            "day_to": 433,
            "ei": None,
            "note": "A Kimi-based Fine-Tuned Leader leads; Claude models follow.",
        },
        {
            "title": "Claude Code agent (near-faithful replay)",
            "day_from": 300,
            "day_to": 360,
            "ei": None,
            "note": "Opus 4.5 (Claude Code): its full native conversation is logged, so asks replay its real context.",
        },
    ]
    moments = [m for m in moments if any(m["day_from"] <= d <= m["day_to"] for d in by_day)]

    room_eis = collections.defaultdict(list)  # room -> sorted chat eis (context-size lookups)
    for r in chat:
        room_eis[r["room"]].append(r["ei"])
    # search haystacks (lowercased once), ei-ordered; per-day slices for the days= pre-filter
    hay = [(r["ei"], (r["content"] or "").lower()) for r in chat]
    hay_think = sorted(
        [(r["ei"], r["thinking"].lower()) for r in chat if r["thinking"]]
        + [(a["ei"], a["thinking"].lower()) for a in acts if a["thinking"]]
    )
    hay_by_day, hay_think_by_day = collections.defaultdict(list), collections.defaultdict(list)
    for e in hay:
        hay_by_day[by_ei[e[0]][1]["day"]].append(e)
    for e in hay_think:
        hay_think_by_day[by_ei[e[0]][1]["day"]].append(e)
    seq_to_ei = collections.defaultdict(list)
    for ei, seq in cc_seq_of_ei.items():
        seq_to_ei[seq].append(ei)
    return {
        "chat": chat,
        "acts": acts,
        "names": names,
        "roster": ros,
        "rows": rows,
        "segs": segs,
        "seg_by_id": {s["seg_id"]: s for s in segs},
        "seg_pos": {s["seg_id"]: i for i, s in enumerate(segs)},
        "seg_starts": [s["start_seq"] for s in segs],
        "seg_turns": seg_turns,
        "cc_glob": cc_glob,
        "by_day": by_day,
        "by_ei": by_ei,
        "cc_seq_of_ei": cc_seq_of_ei,
        "seq_to_ei": seq_to_ei,
        "segs_by_day": segs_by_day,
        "room_eis": room_eis,
        "slug_of": slug_of,
        "sid_of": sid_of,
        "men_m": men_m,
        "men": men,
        "day_men": day_men,
        "men_by": men_by,
        "overview": {
            "day_range": [min(by_day), max(by_day)] if by_day else [0, 0],
            "days": days,
            "eras": eras,
            "agents": agents,
            "rooms": rooms_out,
            "presence": dict(presence),
            "moments": moments,
        },
        "hay": hay,
        "hay_think": hay_think,
        "hay_by_day": hay_by_day,
        "hay_think_by_day": hay_think_by_day,
    }


def _build() -> dict:
    rows = cc.load_all()
    segs = cc.segments(rows)
    return _build_from(tb.load_chat(), tb.load_activity(), tb.room_names(), tb.roster(), rows, segs, cc.day_dates())


def index() -> dict:
    global _IDX
    if _IDX is None:
        with _LOCK:
            if _IDX is None:
                ix = _build()
                _check_days(ix)  # raises before a day-less index is installed (ready() stays False)
                _IDX = ix
    return _IDX


def set_index(ix: Optional[dict]) -> None:
    """Install a prebuilt index (tests)."""
    global _IDX
    _IDX = ix
    _search_cached.cache_clear()


def ready() -> bool:
    return _IDX is not None


def _check_days(ix: dict) -> None:
    """Fail loudly on an index without days; warn about events that fall outside every village day."""
    chat, acts = ix["chat"], ix["acts"]
    if chat and not ix["by_day"]:
        raise RuntimeError(
            f"the village index has 0 days although {len(chat)} chat messages were loaded; "
            "village-transcript.json may not match events.jsonl.gz"
        )
    n_c = sum(r["day"] is None for r in chat)
    n_a = sum(a["day"] is None for a in acts)
    if n_c or n_a:
        print(
            f"  WARNING: {n_c} chat messages and {n_a} activity events fall outside every village day in "
            "village-transcript.json (not on the timeline; is the transcript older than events.jsonl.gz?)",
            file=sys.stderr,
        )


def warm(cc_loaded_cb=None) -> None:
    """Staged load for the server's background thread; progress in STATUS (days -> cc -> chat -> index).
    Any failure, including day ranges that can't be built, ends in stage 'error' with the reason."""
    try:
        # the day ranges first: their one-time transcript parse (~3GB) then doesn't stack on the corpora
        STATUS.update(stage="days", error=None)
        cc.load_days()
        STATUS["stage"] = "cc"
        rows = cc.load_all()
        if cc_loaded_cb:
            cc_loaded_cb(rows)
        STATUS["stage"] = "chat"
        tb.load_chat()
        STATUS["stage"] = "index"
        index()
        STATUS["stage"] = "ready"
    except Exception as e:  # surfaced by /api/status
        STATUS.update(stage="error", error=f"{type(e).__name__}: {e}")
        raise


def status() -> dict:
    out = {"ready": ready(), "stage": STATUS["stage"], "elapsed_s": round(time.time() - STATUS["t0"], 1)}
    if STATUS.get("error"):
        out["error"] = STATUS["error"]
    return out


# --------------------------------------------------------------------------------------------------
# lookups
# --------------------------------------------------------------------------------------------------
def resolve_agent(x: Optional[str], ix: Optional[dict] = None) -> Optional[str]:
    """slug or agent_id -> agent_id (None if empty/unknown)."""
    ix = ix or index()
    if not x:
        return None
    if x in ix["roster"]:
        return x
    return ix["sid_of"].get(x)


def overview() -> dict:
    return index()["overview"]


def _act_summary(a: dict) -> str:
    d = a["detail"]
    t = a["type"]
    if t in ("PAUSE", "WAIT"):
        secs = d.get("seconds")
        return (
            f"paused {int(secs) // 60} min"
            if isinstance(secs, (int, float)) and secs >= 60
            else (f"paused {d.get('duration')}" if d.get("duration") else "paused")
        )
    if t == "CONSOLIDATE":
        return "next session: " + str(d.get("nextShortDisplayedSessionGoal") or d.get("nextSessionGoal") or "")
    if t == "SEARCH_HISTORY":
        return "searched history: " + str(d.get("query") or "")
    if t == "START_USING_COMPUTER":
        g = d.get("sessionGoal") or d.get("goal") or d.get("shortDisplayedSessionGoal") or ""
        return "started using computer" + (f": {g}" if g else "")
    if t == "STOP_USING_COMPUTER":
        return "stopped using computer"
    if t == "ENTER_ROOM":
        return f"entered #{d.get('roomName', '?')}"
    if t == "REQUEST_HUMAN_HELPER":
        return "requested a human helper: " + str(d.get("sessionGoal") or "")
    if t == "CANCEL_REQUEST_FOR_HUMAN_HELPER":
        return "cancelled human-helper request"
    if t == "OUTREACH_APPROVAL_REQUEST":
        return f"asked approval to contact {d.get('recipient', '?')}"
    if t == "OUTREACH_APPROVAL_RESPONSE":
        return f"outreach to {d.get('recipient', '?')}: {'approved' if d.get('approval') else 'declined'}"
    if t == "USER_NAME_CHANGE":
        return "name change"
    return t.replace("_", " ").lower()


def _light(ix: dict, kind: str, r: dict, full: bool = False) -> dict:
    sid = r.get("sid")
    is_h = kind == "c" and not r["is_agent"]
    base = {
        "ei": r["ei"],
        "k": kind,
        "room": _room_label(r.get("room"), ix["names"]),
        "sid": sid,
        "slug": ix["slug_of"].get(sid) if sid and not is_h else None,
        "sp": r.get("speaker"),
        "fam": r.get("family"),
        "fg": "human" if is_h else family_group(r.get("family")),
        "t": _hms(r["created_at"]),
        "th": r.get("think_kind") or ("x" if r.get("thinking") else ""),
    }
    if kind == "c":
        c = r["content"] or ""
        base.update(
            {
                "h": is_h,
                "hc": _hc(r),
                "c": c if full else c[:TALK_PREVIEW],
                "more": False if full else len(c) > TALK_PREVIEW,
                "p": ("cc" if r["ei"] in ix["cc_seq_of_ei"] else "tierb") if r["is_agent"] else "",
                "n_probes": 0,
                "star": False,
            }
        )
    else:
        base.update({"type": r["type"], "s": _act_summary(r)[:240]})
    return base


def talked_about(ix: dict, day_n: int, top: int = 8) -> list:
    ros = ix["roster"]
    return [
        {
            "slug": ix["slug_of"][sid],
            "name": ros[sid]["name"],
            "family": ros[sid]["family"],
            "family_group": family_group(ros[sid]["family"]),
            "n": n,
        }
        for sid, n in ix["day_men"].get(day_n, collections.Counter()).most_common(top)
    ]


def day(n: int, full: bool = False) -> dict:
    """Every event of village day n. full=True ships complete message text (the timeline shows
    messages expanded); the default ships TALK_PREVIEW-char previews."""
    ix = index()
    evs = ix["by_day"].get(n)
    if evs is None:
        act = sorted(ix["by_day"])
        i = bisect.bisect_left(act, n)
        prev_d = act[i - 1] if i > 0 else None
        next_d = act[i] if i < len(act) else None
        raise ValueError(f"no activity on Day {n} (nearest active: {prev_d} / {next_d})")
    ov = next(x for x in ix["overview"]["days"] if x["day"] == n)
    era = ix["overview"]["eras"][ov["era"]] if ov.get("era") is not None else None
    segs = [
        {k: s[k] for k in ("seg_id", "start", "end", "n_turns", "boundary", "preview")}
        for s in ix["segs_by_day"].get(n, [])
    ]
    return {
        "day": n,
        "date": ov["date"],
        "stats": ov,
        "era": era,
        "events": [_light(ix, k, r, full) for k, r in evs],
        "cc_segments": segs,
        "talked_about": talked_about(ix, n),
    }


def _turn_for_seq(ix: dict, seg_id: int, seq: int) -> Optional[dict]:
    """The turn containing seq (first_seq <= seq <= last_seq) in that segment, or None."""
    for t in ix["seg_turns"].get(seg_id, []):
        if t["first_seq"] <= seq <= t["last_seq"]:
            return t
    return None


def _seg_for_seq(ix: dict, seq: int) -> Optional[dict]:
    i = bisect.bisect_right(ix["seg_starts"], seq) - 1
    if i < 0:
        return None
    seg = ix["segs"][i]
    return seg if seg["start_seq"] <= seq <= seg["end_seq"] else None


def cc_link(ei: int) -> Optional[dict]:
    """A Claude Code agent chat message -> {seg_id, turn, anchor_before, anchor_after, boundary,
    n_turns}: the native turn that issued the mcp__village__chat_message call carrying it."""
    ix = index()
    seq = ix["cc_seq_of_ei"].get(ei)
    if seq is None:
        return None
    seg = _seg_for_seq(ix, seq)
    if seg is None:
        return None
    t = _turn_for_seq(ix, seg["seg_id"], seq)
    if t is None:
        return None
    return {
        "seg_id": seg["seg_id"],
        "turn": t["turn"],
        "anchor_before": t["anchor_before"],
        "anchor_after": t["anchor_after"],
        "boundary": seg["boundary"],
        "n_turns": seg["n_turns"],
    }


def cc_at(ei: int) -> Optional[dict]:
    """Asking the CC agent AT someone else's message: its last COMPLETED turn at/before the message
    time (within CC_TIME_MATCH_MAX_S). -> {seg_id, seq (= turn.anchor_after), turn, gap_s,
    last_read:{turn, t}|None}  (last_read = last turn <= it in the window that ran get_events)."""
    ix = index()
    hit = ix["by_ei"].get(ei)
    if hit is None:
        raise ValueError("unknown event_index")
    t = tb.parse_ts(hit[1]["created_at"])
    if t is None or not ix["cc_glob"]:
        return None
    ts = t.timestamp()
    i = bisect.bisect_right(ix["cc_glob"], (ts, float("inf"), float("inf"))) - 1
    if i < 0:
        return None
    t_end, seg_id, turn_i = ix["cc_glob"][i]
    gap = ts - t_end
    if gap > CC_TIME_MATCH_MAX_S:
        return None
    turns = ix["seg_turns"][seg_id]
    turn = turns[turn_i]
    last_read = next(({"turn": x["turn"], "t": x["t_end"]} for x in reversed(turns[: turn_i + 1]) if x["reads"]), None)
    return {
        "seg_id": seg_id,
        "seq": turn["anchor_after"],
        "turn": turn_i,
        "gap_s": round(gap, 1),
        "last_read": last_read,
    }


def _probe_info(ix: dict, kind: str, r: dict) -> Optional[dict]:
    """How this event can be probed, or None (humans, activity)."""
    if kind != "c" or not r["is_agent"]:
        return None
    if r["ei"] in ix["cc_seq_of_ei"]:
        link = cc_link(r["ei"])
        if link:
            return {"tier": "cc", **link}
    return {"tier": "tierb", "agent": ix["slug_of"].get(r["sid"]), "agent_id": r["sid"], "ei": r["ei"]}


def event(ei: int) -> dict:
    ix = index()
    hit = ix["by_ei"].get(ei)
    if hit is None:
        raise ValueError("unknown event_index")
    kind, r = hit
    out = _light(ix, kind, r)
    out.update(
        {
            "day": r["day"],
            "created_at": r["created_at"],
            "thinking": r.get("thinking") or "",
            "think_kind": r.get("think_kind") or "",
        }
    )
    if kind == "c":
        out["c"] = r["content"] or ""
        out["more"] = False
        out["n_prior_in_room"] = bisect.bisect_left(ix["room_eis"][r["room"]], ei)
        out["mentions"] = [ix["slug_of"][s] for s in ix["men"].get(ei, ())]
    else:
        out["detail"] = r["detail"]
    out["probe"] = _probe_info(ix, kind, r)
    return out


def agent_series(agent: str) -> dict:
    ix = index()
    sid = resolve_agent(agent, ix)
    if sid is None:
        raise ValueError("unknown agent")
    slug = ix["slug_of"][sid]
    return {
        "slug": slug,
        "agent_id": sid,
        "aliases": sorted(ix["men_m"]["aliases_by_sid"].get(sid, [])),
        "own": ix["overview"]["presence"].get(slug, {}),
        "mentioned_by_others": dict(ix["men_by"].get(sid, {})),
        "approx": True,
    }


def cc_window_resumed(seg: dict) -> bool:
    """The window begins where Claude Code resumed its session (an `init` after earlier rows)."""
    return seg.get("boundary") == "init" and seg.get("start_seq", 0) > 0


def cc_window_fidelity(seg: dict) -> str:
    tail = (
        " The exact CC system prompt is not in the dataset; a flagged stand-in is used. Prior in-turn "
        "thinking is shown for browsing but not fed back into the reconstructed context."
    )
    if seg.get("boundary") == "compact":
        if seg.get("has_summary", True):
            head = (
                "Near-faithful replay of the agent's real turns in this window. This window begins at a "
                "Claude Code context compaction: it opens with Claude Code's own summary of the earlier "
                "context, which is in the dataset and is replayed as the first message, just as the agent "
                "had it after the compaction."
            )
        else:
            head = (
                "Near-faithful replay of the agent's real turns in this window. This window begins at a "
                "Claude Code context compaction, but its compaction summary is not in the dataset, so the "
                "agent knew more than shown here."
            )
    elif cc_window_resumed(seg):
        head = (
            "Near-faithful replay of the agent's real turns in this window. This window begins where "
            "Claude Code resumed its session; the agent usually kept its earlier context across a resume, "
            "and that earlier context is not replayed here, so it knew more than shown."
        )
    else:
        head = (
            "Near-faithful replay of the agent's real turns in this window, from the start of its Claude "
            "Code logs (there is no earlier context)."
        )
    return head + tail


def cc_window(seg_id: int) -> dict:
    ix = index()
    seg = ix["seg_by_id"].get(seg_id)
    if seg is None:
        raise ValueError("unknown seg_id")
    pos = ix["seg_pos"][seg_id]
    turns = cc.timeline(cc.segment_slice(ix["rows"], seg))
    for t in turns:
        t["chat_ei"] = sorted(ei for q in range(t["first_seq"], t["last_seq"] + 1) for ei in ix["seq_to_ei"].get(q, ()))
    summary = dict(seg)
    summary.update(
        {
            "fidelity": cc_window_fidelity(seg),
            "resumed": cc_window_resumed(seg),
            "agent": "Opus 4.5 (Claude Code)",
            "prev_seg_id": ix["segs"][pos - 1]["seg_id"] if pos > 0 else None,
            "next_seg_id": ix["segs"][pos + 1]["seg_id"] if pos + 1 < len(ix["segs"]) else None,
        }
    )
    return {"summary": summary, "turns": turns}


# --------------------------------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------------------------------
_TOK = re.compile(r'(\w+):"([^"]*)"|(\w+):(\S+)|"([^"]+)"|(\S+)')
_FAMILIES = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "xai": "xAI",
    "deepseek": "DeepSeek",
    "moonshot": "Moonshot",
    "zhipu": "Zhipu",
    "meta": "Meta",
}


def parse_query(q: str, ix: Optional[dict] = None) -> dict:
    """Grammar: words AND, "quoted phrases", from:<slug|family|other|human|operator>, about:<slug>,
    room:<name>, day:447 | day:440-452, in:thinking, kind:agent|human|operator. A bare 447 / d447 /
    2026-06-23 / ei:NNN is a JUMP."""
    ix = ix or index()
    q = (q or "").strip()
    out = {
        "terms": [],
        "from": None,
        "about": None,
        "room": None,
        "days": None,
        "thinking": False,
        "kind": None,
        "jump": None,
    }
    m = re.fullmatch(r"d?(\d{1,3})", q, re.I)
    if m:
        out["jump"] = {"day": int(m.group(1))}
        return out
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", q):
        # several village days can share a date (a short stub before the real session): take the busiest
        same = [x for x in ix["overview"]["days"] if x["date"] == q]
        if not same:
            raise ValueError(f"no village activity on {q}")
        out["jump"] = {"day": max(same, key=lambda x: x["n_msgs"] + x["n_act"])["day"]}
        return out
    m = re.fullmatch(r"ei:(\d+)", q, re.I)
    if m:
        hit = ix["by_ei"].get(int(m.group(1)))
        if hit is None:
            raise ValueError("unknown event_index")
        out["jump"] = {"day": hit[1]["day"], "ei": int(m.group(1))}
        return out
    for k1, v1, k2, v2, phrase, word in _TOK.findall(q):
        key, val = (k1, v1) if k1 else (k2, v2)
        key = key.lower()
        if key == "from":
            v = val.lower()
            sid = resolve_agent(val, ix)
            if sid:
                out["from"] = ("agent", sid)
            elif v in _FAMILIES:
                out["from"] = ("family", _FAMILIES[v])
            elif v in ("other", "human", "operator"):
                out["from"] = (v, None)
            else:
                raise ValueError(f"unknown from: {val!r} (use an agent slug, a family, other, human or operator)")
        elif key == "about":
            sid = resolve_agent(val, ix)
            if not sid:
                raise ValueError(f"unknown about: {val!r} (use an agent slug)")
            out["about"] = sid
        elif key == "room":
            out["room"] = val.lstrip("#")
        elif key == "day":
            mm = re.fullmatch(r"(\d+)(?:-(\d+))?", val)
            if not mm:
                raise ValueError(f"bad day: {val!r} (day:447 or day:440-452)")
            a = int(mm.group(1))
            out["days"] = (a, int(mm.group(2) or a))
        elif key == "in":
            if val.lower() != "thinking":
                raise ValueError("in: only supports in:thinking")
            out["thinking"] = True
        elif key == "kind":
            if val.lower() not in ("agent", "human", "operator"):
                raise ValueError("kind: must be agent, human or operator")
            out["kind"] = val.lower()
        elif key:  # unknown key: treat 'x:y' as a literal term
            out["terms"].append(f"{key}:{val}".lower())
        else:
            out["terms"].append((phrase or word).lower())
    return out


def _passes(ix: dict, f: dict, kind: str, r: dict) -> bool:
    is_h = kind == "c" and not r["is_agent"]
    fr = f["from"]
    if fr:
        typ, v = fr
        if typ == "agent" and (is_h or r.get("sid") != v):
            return False
        if typ == "family" and (is_h or r.get("family") != v):
            return False
        if typ == "other" and (is_h or family_group(r.get("family")) != "other"):
            return False
        if typ == "human" and not is_h:
            return False
        if typ == "operator" and not (is_h and r["speaker"] in OPERATORS):
            return False
    if f["kind"]:
        if f["kind"] == "agent" and is_h:
            return False
        if f["kind"] == "human" and not is_h:
            return False
        if f["kind"] == "operator" and not (is_h and r["speaker"] in OPERATORS):
            return False
    if f["room"] and _room_label(r.get("room"), ix["names"]).lower() != f["room"].lower():
        return False
    if f["about"] and f["about"] not in ix["men"].get(r["ei"], ()):
        return False
    return True


def search(q: str, *, days: Optional[tuple] = None, limit: int = 500, offset: int = 0) -> dict:
    return _search_cached(q or "", tuple(days) if days else None, int(limit), int(offset))


@functools.lru_cache(maxsize=20)
def _search_cached(q: str, days: Optional[tuple], limit: int, offset: int) -> dict:
    ix = index()
    f = parse_query(q, ix)
    filters = {}
    if f["from"]:
        typ, v = f["from"]
        filters["from"] = [typ, ix["slug_of"].get(v, v) if typ == "agent" else v]
    if f["about"]:
        filters["about"] = ix["slug_of"].get(f["about"])
    for k in ("room", "kind"):
        if f[k]:
            filters[k] = f[k]
    if f["days"]:
        filters["days"] = list(f["days"])
    base = {
        "q": q,
        "terms": f["terms"],
        "thinking": f["thinking"],
        "filters": filters,
        "total": 0,
        "per_day": {},
        "hits": [],
        "truncated": False,
    }
    if f["jump"]:
        base["jump"] = f["jump"]
        return base
    if not f["terms"] and not any(f[k] for k in ("from", "about", "room", "kind")):
        raise ValueError("empty query")
    rng = f["days"] or days
    if f["days"] and days:  # intersect the query's day: with days=
        rng = (max(f["days"][0], days[0]), min(f["days"][1], days[1]))
    if rng:
        src = ix["hay_think_by_day"] if f["thinking"] else ix["hay_by_day"]
        hay = [e for d in sorted(k for k in src if k is not None) if rng[0] <= d <= rng[1] for e in src[d]]
    else:
        hay = ix["hay_think"] if f["thinking"] else ix["hay"]
    terms = f["terms"]
    per_day = collections.Counter()
    hits = []
    total = 0
    for ei, text in hay:
        if terms and not all(t in text for t in terms):
            continue
        kind, r = ix["by_ei"][ei]
        if not _passes(ix, f, kind, r):
            continue
        total += 1
        if r["day"] is not None:
            per_day[r["day"]] += 1
        if offset < total <= offset + limit:
            src_text = (r.get("thinking") if f["thinking"] else r.get("content")) or ""
            pos = max(src_text.lower().find(terms[0]) if terms else 0, 0)
            a = max(0, pos - 80)
            snip = (
                ("…" if a else "")
                + src_text[a : pos + 160].replace("\n", " ")
                + ("…" if pos + 160 < len(src_text) else "")
            )
            h = _light(ix, kind, r)
            h.update({"day": r["day"], "snip": snip})
            hits.append(h)
    base.update({"total": total, "per_day": dict(per_day), "hits": hits, "truncated": total > offset + len(hits)})
    return base


# --------------------------------------------------------------------------------------------------
# unified probe prompt (no model call) + the live call
# --------------------------------------------------------------------------------------------------
def prompt_sha(system: str, messages: list) -> str:
    return hashlib.sha256(
        json.dumps({"system": system, "messages": messages}, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _approx_tokens(system: str, messages: list) -> int:
    return (len(system) + sum(len(m.get("text", "")) for m in messages)) // 4


def normalize_params(p: dict) -> dict:
    """Coerce request params (query-string or JSON) into typed probe params."""

    def _int(k, default=None):
        v = p.get(k)
        if v in (None, ""):
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            raise ValueError(f"{k} must be an integer")

    target = (p.get("target") or "tierb").lower()
    if target not in ("tierb", "cc"):
        raise ValueError("target must be 'tierb' or 'cc'")
    mode = (p.get("mode") or "belief").lower()
    if mode not in ("belief", "resample"):
        raise ValueError("mode must be 'belief' or 'resample'")
    anchor = (p.get("anchor") or "").lower() or None
    if anchor not in (None, "before", "after"):
        raise ValueError("anchor must be 'before' or 'after'")
    window = _int("window", 80)
    if not 1 <= window <= 400:
        raise ValueError("window must be 1..400")
    fmt = (p.get("format") or "turns").lower()
    if fmt not in tb.PROMPT_FORMATS:
        raise ValueError("format must be 'transcript' or 'turns'")
    model = p.get("model") or DEFAULT_MODEL
    if not isinstance(model, str) or llm.provider_for(model) is None:
        raise ValueError(llm.unroutable_reason(str(model)))
    return {
        "target": target,
        "agent": p.get("agent") or None,
        "ei": _int("ei"),
        "anchor": anchor,
        "window": window,
        "format": fmt,
        "mode": mode,
        "question": p.get("question") or "",
        "seg_id": _int("seg_id"),
        "seq": _int("seq"),
        "at_ei": _int("at_ei"),
        "model": model,
    }


def _model_mismatch(model: str, actual_model: str, name: str) -> Optional[dict]:
    if tb.same_model(model, actual_model):  # the same exact-match rule as the approximate path
        return None
    return {
        "code": "model_mismatch",
        "severity": "warn",
        "text": f"Voiced by {model} · original: {name}" + (f" ({actual_model})" if actual_model else ""),
    }


def probe_prompt(params: dict) -> dict:
    """The exact prompt a probe with these params will send, plus how the anchor resolved, the
    context it covers, the fidelity block and the 'actual' to compare against. No model call.
    params: see normalize_params()."""
    p = normalize_params(params)
    ix = index()
    return _probe_prompt_tierb(ix, p) if p["target"] == "tierb" else _probe_prompt_cc(ix, p)


def _probe_prompt_tierb(ix: dict, p: dict) -> dict:
    ei = p["ei"]
    if ei is None:
        raise ValueError("ei is required for target=tierb")
    hit = ix["by_ei"].get(ei)
    if hit is None:
        raise ValueError("unknown event_index")
    kind, tgt = hit
    if kind != "c":
        raise ValueError("activity events can't be probed — pick a chat message")
    agent_id = resolve_agent(p["agent"], ix) if p["agent"] else (tgt["sid"] if tgt["is_agent"] else None)
    if agent_id is None:
        raise ValueError("unknown agent" if p["agent"] else "pick an agent to ask about a human's message")
    is_author = tgt["is_agent"] and agent_id == tgt["sid"]
    anchor = (p["anchor"] or "before") if is_author else "after"
    chat, acts = ix["chat"], ix["acts"]
    pp = tb.build_probe_prompt(
        agent_id, ei, p["question"], mode=p["mode"], window=p["window"], chat=chat, anchor=anchor, fmt=p["format"]
    )
    oc = tb.observable_context(ei, chat, p["window"], include_target=(anchor == "after"))
    msgs_ctx = oc["messages"]
    fid = tb.fidelity(
        agent_id, ei, anchor, p["window"], p["model"], n_context=len(msgs_ctx), chat=chat, acts=acts, fmt=p["format"]
    )
    fid.pop("presence", None)
    if agent_id == cc.CC_AGENT_ID and ei in ix["cc_seq_of_ei"]:
        fid["warnings"].append(
            {
                "code": "cc_available",
                "severity": "info",
                "text": "A near-faithful replay of its real Claude Code context exists for this "
                "message (target=cc) — this is the approximate room-chat reconstruction.",
            }
        )
    elif agent_id == cc.CC_AGENT_ID and is_author:
        fid["warnings"].append(
            {
                "code": "cc_unmapped",
                "severity": "info",
                "text": "This Claude Code message could not be matched to its native turn, so the "
                "approximate room-chat reconstruction is used.",
            }
        )
    act = tb.actual_for(agent_id, ei, anchor, chat)
    act.pop("created_at", None)
    act.update({"turn": None, "actions": None})
    name = ix["roster"].get(agent_id, {}).get("name") or "this agent"
    if is_author:
        desc = "just before its message" if anchor == "before" else "right after its message"
    else:
        desc = f"right after {tgt['speaker']}'s message"
    tday = tgt["day"]
    earlier = {m["day"] for m in msgs_ctx if m["day"] is not None and tday is not None and m["day"] < tday}
    return {
        "system": pp["system"],
        "messages": pp["messages"],
        "mode": p["mode"],
        "resample_ok": True,
        "anchor_desc": desc,
        "resolved": {
            "target": "tierb",
            "agent": ix["slug_of"].get(agent_id),
            "agent_id": agent_id,
            "agent_name": name,
            "ei": ei,
            "anchor": anchor,
            "seg_id": None,
            "seq": None,
            "turn": None,
            "match": None,
            "gap_s": None,
            "day": tday,
            "room": _room_label(tgt["room"], ix["names"]),
        },
        "context": {
            "room": _room_label(oc["room"], ix["names"]),
            "first_ei": msgs_ctx[0]["ei"] if msgs_ctx else None,
            "last_ei": msgs_ctx[-1]["ei"] if msgs_ctx else None,
            "n": len(msgs_ctx),
            "t0": msgs_ctx[0]["created_at"] if msgs_ctx else None,
            "t1": msgs_ctx[-1]["created_at"] if msgs_ctx else None,
            "speakers": sorted({m["speaker"] for m in msgs_ctx}),
            "approx_tokens": _approx_tokens(pp["system"], pp["messages"]),
            "n_earlier_days": len(earlier),
        },
        "fidelity": fid,
        "actual": act,
        "default_question": tb._RESAMPLE_Q if p["mode"] == "resample" else TIERB_DEFAULT_Q,
        "sha256": prompt_sha(pp["system"], pp["messages"]),
        "window": p["window"],
        "format": p["format"],
        "model": p["model"],
    }


def _cc_turn_actual(ix: dict, seg_id: int, turn_i: Optional[int], kind: str) -> dict:
    none = {
        "kind": "none",
        "ei": None,
        "turn": None,
        "content": "",
        "thinking": "",
        "think_kind": "",
        "gap_s": None,
        "actions": None,
    }
    turns = ix["seg_turns"].get(seg_id, [])
    if turn_i is None or not 0 <= turn_i < len(turns):
        return none
    t = turns[turn_i]
    full = next(
        (x for x in cc.timeline(cc.segment_slice(ix["rows"], ix["seg_by_id"][seg_id])) if x["turn"] == turn_i), None
    )
    if full is None:
        return none
    eis = sorted(ei for q in range(t["first_seq"], t["last_seq"] + 1) for ei in ix["seq_to_ei"].get(q, ()))
    return {
        "kind": kind,
        "ei": eis[0] if eis else None,
        "turn": turn_i,
        "content": full["text"],
        "thinking": full["thinking"],
        "think_kind": "verbatim" if full["thinking"] else "",
        "gap_s": None,
        "actions": full["actions"],
    }


def _probe_prompt_cc(ix: dict, p: dict) -> dict:
    warnings, match, gap_s, ei, tgt, ref_turn = [], None, None, None, None, None
    at = p["at_ei"] if p["at_ei"] is not None else (p["ei"] if p["seg_id"] is None else None)
    if p["seg_id"] is not None:  # explicit window + seq (the CC track's gaps)
        seg = ix["seg_by_id"].get(p["seg_id"])
        if seg is None:
            raise ValueError("unknown seg_id")
        if p["seq"] is None:
            raise ValueError("seq is required with seg_id")
        seq = p["seq"]
        if not seg["start_seq"] - 1 <= seq <= seg["end_seq"]:
            raise ValueError("seq is outside that window")
        turns = ix["seg_turns"][seg["seg_id"]]
        nxt = next((t for t in turns if t["first_seq"] > seq), None)
        done = [t for t in turns if t["last_seq"] <= seq]
        turn_i = done[-1]["turn"] if done else None
        at_gap = nxt is not None and nxt["anchor_before"] == seq
        if at_gap:
            desc = f"at the gap before turn #{nxt['turn']}" + (
                f" (after turn #{turn_i}'s results)" if turn_i is not None else ""
            )
        elif turn_i is not None:
            desc = f"after turn #{turn_i}"
        else:
            desc = "at the start of the window"
        anchor = "before" if at_gap else "after"
        ref_turn = nxt["turn"] if at_gap else turn_i
        actual = _cc_turn_actual(ix, seg["seg_id"], nxt["turn"] if nxt else None, "next")
    elif at is not None:
        hit = ix["by_ei"].get(at)
        if hit is None:
            raise ValueError("unknown event_index")
        _, tgt = hit
        ei = at
        if at in ix["cc_seq_of_ei"]:  # its OWN message -> the posting turn
            link = cc_link(at)
            if link is None:
                raise ValueError("this Claude Code message could not be mapped to a turn")
            seg = ix["seg_by_id"][link["seg_id"]]
            anchor = p["anchor"] or "before"
            match, ref_turn = "posting_turn", link["turn"]
            if anchor == "before":
                seq = link["anchor_before"]
                desc = f"just before turn #{link['turn']} (the turn that posted this message)"
                actual = _cc_turn_actual(ix, seg["seg_id"], link["turn"], "this")
            else:
                seq = link["anchor_after"]
                desc = f"right after turn #{link['turn']} (the turn that posted this message)"
                actual = _cc_turn_actual(ix, seg["seg_id"], link["turn"] + 1, "next")
        else:  # someone else's message -> matched by time
            m = cc_at(at)
            if m is None:
                raise ValueError("the Claude Code agent had no completed turn within 6h before this message")
            seg = ix["seg_by_id"][m["seg_id"]]
            seq, gap_s, match, anchor, ref_turn = m["seq"], m["gap_s"], "time", "after", m["turn"]
            mins = max(1, round(gap_s / 60))
            desc = f"after turn #{m['turn']} · matched by time ({mins} min before this message)"
            actual = _cc_turn_actual(ix, seg["seg_id"], m["turn"] + 1, "next")
            speaker = tgt.get("speaker") or "?"
            warnings.append(
                {
                    "code": "cross_anchor",
                    "severity": "info",
                    "text": f"Asking Opus 4.5 (Claude Code) at {speaker}'s message",
                }
            )
            warnings.append(
                {
                    "code": "cc_time_match",
                    "severity": "warn" if gap_s > 600 else "info",
                    "text": f"matched by time · its last completed turn was {mins} min before this message",
                }
            )
            lr = m["last_read"]
            warnings.append(
                {
                    "code": "cc_unread",
                    "severity": "warn",
                    "text": (
                        (
                            f"its last village read (get_events) in this window was turn #{lr['turn']} at "
                            f"{(lr['t'] or '')[11:16]}; later chat is not in its context"
                        )
                        if lr
                        else "it made no village read (get_events) in this window before this point; "
                        "the room chat is not in its context"
                    ),
                }
            )
    else:
        raise ValueError("target=cc needs seg_id+seq or at_ei")
    sl = cc.segment_slice(ix["rows"], seg)
    resample_ok = cc._ends_on_user(cc.build_messages(sl, seq))
    if p["mode"] == "resample" and not resample_ok:
        raise ValueError(
            "resample needs the context to end on a tool result (a decision point); this anchor "
            "ends on the agent's own turn"
        )
    pp = cc.build_probe_prompt(sl, seq, p["question"], mode=p["mode"])
    turns = ix["seg_turns"][seg["seg_id"]]
    done = [t for t in turns if t["last_seq"] <= seq]
    mm = _model_mismatch(p["model"], cc.CC_MODEL_ACTUAL, "Opus 4.5 (Claude Code)")
    if mm:
        warnings.insert(0, mm)
    missing = [
        "the exact Claude Code system prompt (a flagged stand-in is used)",
        "earlier in-turn thinking (shown for browsing, not replayed)",
    ]
    if cc_window_resumed(seg):
        missing.insert(1, "the earlier context of this resumed session (not replayed)")
    elif seg["boundary"] == "compact" and not seg.get("has_summary", True):
        missing.insert(1, "the compaction summary this window began with (not in the dataset)")
    k = done[-1]["turn"] if done else None
    fid = {
        "level": "near_faithful",
        "badge": "NEAR-FAITHFUL",
        "summary": (
            f"Replays its real Claude Code window #{seg['seg_id']} through turn #{k}."
            if k is not None
            else f"Replays its real Claude Code window #{seg['seg_id']} from the start."
        ),
        "missing": missing,
        "warnings": warnings,
        "counterfactual": False,
        "details": cc_window_fidelity(seg),
    }
    row_t = next((r.created_at for r in reversed(sl) if r.seq <= seq), seg["start"])
    day_n = tgt["day"] if tgt is not None else seg["day"]
    return {
        "system": pp["system"],
        "messages": pp["messages"],
        "mode": p["mode"],
        "resample_ok": resample_ok,
        "anchor_desc": desc,
        "resolved": {
            "target": "cc",
            "agent": ix["slug_of"].get(cc.CC_AGENT_ID),
            "agent_id": cc.CC_AGENT_ID,
            "agent_name": "Opus 4.5 (Claude Code)",
            "ei": ei,
            "anchor": anchor,
            "seg_id": seg["seg_id"],
            "seq": seq,
            "turn": ref_turn,
            "turns_done": k,
            "match": match,
            "gap_s": gap_s,
            "day": day_n,
            "room": _room_label(tgt["room"], ix["names"]) if tgt is not None else None,
        },
        "context": {
            "room": None,
            "first_ei": None,
            "last_ei": None,
            "n": len(done),
            "t0": seg["start"],
            "t1": row_t,
            "speakers": [],
            "approx_tokens": _approx_tokens(pp["system"], pp["messages"]),
            "n_earlier_days": 0,
            "seg_id": seg["seg_id"],
            "turn_range": [0, k] if k is not None else None,
        },
        "fidelity": fid,
        "actual": actual,
        "default_question": "" if p["mode"] == "resample" else CC_DEFAULT_Q,
        "sha256": prompt_sha(pp["system"], pp["messages"]),
        "window": None,
        "format": None,  # the Claude Code replay is always its native turns
        "model": p["model"],
    }


def call_model(system: str, messages: list, model: str) -> dict:
    """The live model call (provider inferred from the model id; see llm.call)."""
    return llm.call(system, messages, model)

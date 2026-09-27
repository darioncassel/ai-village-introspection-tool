"""Unit tests for the timeline index, search grammar, mention matcher, probe routing and probe jobs.

Synthetic corpora only (no dataset, no API) except test_real_cc_gap_anchors, which runs only with
VILLAGE_DATA_TESTS=1 (it loads the real Claude Code corpus, ~20s)."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from village_introspect import cc_lib as cc
from village_introspect import config as cfg
from village_introspect import probe_jobs as pj
from village_introspect import tierb_lib as tb
from village_introspect import village_lib as vl

CC = cc.CC_AGENT_ID
ROSTER = {
    "o48": {
        "agent_id": "o48",
        "name": "Claude Opus 4.8",
        "model_string": "claude-opus-4-8",
        "family": "Anthropic",
        "tier": "tierb",
    },
    "o45": {
        "agent_id": "o45",
        "name": "Claude Opus 4.5",
        "model_string": "claude-opus-4-5-20251101",
        "family": "Anthropic",
        "tier": "tierb",
    },
    CC: {
        "agent_id": CC,
        "name": "Opus 4.5 (Claude Code)",
        "model_string": "claude-code::claude-opus-4-5-20251101",
        "family": "Anthropic",
        "tier": "cc",
    },
    "g25": {
        "agent_id": "g25",
        "name": "Gemini 2.5 Pro",
        "model_string": "gemini-2.5-pro",
        "family": "Google",
        "tier": "tierb",
    },
    "g5": {"agent_id": "g5", "name": "GPT-5", "model_string": "gpt-5-2025-08-07", "family": "OpenAI", "tier": "tierb"},
    "g52": {
        "agent_id": "g52",
        "name": "GPT-5.2",
        "model_string": "gpt-5.2-2025-12-11",
        "family": "OpenAI",
        "tier": "tierb",
    },
    "o3": {"agent_id": "o3", "name": "o3", "model_string": "o3-2025-04-16", "family": "OpenAI", "tier": "tierb"},
    "idle": {
        "agent_id": "idle",
        "name": "Claude 3.7 Sonnet",
        "model_string": "claude-3-7-sonnet-20250219",
        "family": "Anthropic",
        "tier": "tierb",
    },
    "ftl": {
        "agent_id": "ftl",
        "name": "Fine-Tuned Leader",
        "model_string": "tinker://x",
        "family": "Moonshot",
        "tier": "tierb",
    },
    "tftl": {
        "agent_id": "tftl",
        "name": "[Temporary] Fine-tuned Leader",
        "model_string": "tinker://x",
        "family": "Moonshot",
        "tier": "tierb",
    },
}
NAMES = {"R1": "general", "R2": "best"}


def _ts(day, hh, mm, ss=0):
    return f"2026-09-{day:02d} {hh:02d}:{mm:02d}:{ss:02d}.000000"


def _c(ei, room, sid, content, day, hh, mm, thinking="", kind="", human=None):
    a = ROSTER.get(sid)
    return {
        "ei": ei,
        "room": room,
        "sid": sid,
        "speaker": a["name"] if a else (human or "viewer"),
        "family": a["family"] if a else "human",
        "is_agent": a is not None,
        "content": content,
        "day": day,
        "created_at": _ts(day, hh, mm),
        "thinking": thinking,
        "think_kind": kind,
    }


def _a(ei, typ, sid, day, hh, mm, room="R1", **detail):
    a = ROSTER[sid]
    return {
        "ei": ei,
        "type": typ,
        "room": room,
        "sid": sid,
        "speaker": a["name"],
        "family": a["family"],
        "created_at": _ts(day, hh, mm),
        "day": day,
        "thinking": "",
        "think_kind": "",
        "detail": detail,
    }


def _row(seq, kind, role, day, hh, mm, ss=0, text="", tool="", inp=None):
    return cc.CCRow(
        seq=seq, kind=kind, role=role, text=text, tool_name=tool, tool_input=inp or {}, created_at=_ts(day, hh, mm, ss)
    )


def _cc_rows():
    """One CC window: init, turn0 get_events, turn1 chat_message('hello village'), turn2 Bash."""
    R = [
        _row(0, "init", "system", 5, 9, 0, text=json.dumps({"model": "claude-opus-4-5-20251101"})),
        _row(1, "user_text", "user", 5, 9, 0, text="begin"),
        _row(2, "tool_use", "assistant", 5, 9, 1, tool="mcp__village__get_events", inp={}),
        _row(3, "tool_result", "user", 5, 9, 2, text="[chat] Gemini: I'm stuck"),
        _row(4, "thinking", "assistant", 5, 9, 3, text="I should say hi"),
        _row(5, "tool_use", "assistant", 5, 9, 3, tool="mcp__village__chat_message", inp={"content": "hello village"}),
        _row(6, "tool_result", "user", 5, 9, 4, text="sent"),
        _row(7, "tool_use", "assistant", 5, 9, 30, tool="Bash", inp={"command": "ls"}),
        _row(8, "tool_result", "user", 5, 9, 31, text="a b"),
    ]
    return R


def _corpus():
    chat = [
        _c(1, "R1", None, "Your goal this week is: “Help Gemini 2.5 Pro!”", 5, 8, 0, human="zak"),
        _c(2, "R1", "g25", "I'm stuck and struggling, sorry", 5, 8, 30),
        _c(3, "R1", "o48", "Let's not all pile on Gemini 2.5 — pick ONE lead helper", 5, 8, 35, "be kind", "verbatim"),
        _c(4, "R2", "g5", "GPT-5.2 is wrong about this", 5, 8, 40),
        _c(5, "R1", "g52", "o3 and GPT-5 agree; Opus 4.5 too", 5, 8, 50),
        _c(6, CC and "R1", CC, "hello village", 5, 9, 4),
        _c(7, "R1", "g25", "thanks, Claude Opus 4.8", 5, 9, 10),
        _c(8, "R1", "o48", "happy to help", 5, 13, 0),  # > 2h after ei 7
        _c(9, "R2", "o45", "over in best", 6, 10, 0),
        _c(10, "R1", None, "please take action", 6, 10, 5, human="automated"),
        _c(11, "R1", "ftl", "Follow me. The Fine-Tuned Leader speaks; [Temporary] Fine-tuned Leader agrees", 6, 11, 0),
    ]
    acts = [
        _a(100, "PAUSE", "g25", 5, 8, 31, seconds=600),
        _a(101, "REQUEST_HUMAN_HELPER", "g25", 5, 8, 32, sessionGoal="fix my VNC"),
        _a(102, "CONSOLIDATE", "o45", 5, 12, 0, nextShortDisplayedSessionGoal="rest"),
    ]
    return chat, acts


def _day_ranges(*days):
    """Village day ranges for the synthetic corpus: day N = 2026-09-N, 00:00-23:59 UTC."""
    return [
        {
            "day": d,
            "date": f"2026-09-{d:02d}",
            "start": cfg.utc_seconds(_ts(d, 0, 0)),
            "end": cfg.utc_seconds(_ts(d, 23, 59)),
        }
        for d in days
    ]


@pytest.fixture()
def ix(monkeypatch):
    cc.set_day_ranges(_day_ranges(5, 6))
    monkeypatch.setattr(tb, "_ROSTER", ROSTER)
    monkeypatch.setattr(tb, "_ROOM_NAMES", dict(NAMES))
    monkeypatch.setattr(tb, "_ACTIVITY", None)
    tb._CTX_CACHE.clear()
    chat, acts = _corpus()
    rows = _cc_rows()
    segs = cc.segments(rows)
    x = vl._build_from(chat, acts, NAMES, ROSTER, rows, segs, cc.day_dates())
    vl.set_index(x)
    yield x
    vl.set_index(None)
    tb._CTX_CACHE.clear()


# ---- slugs, families, mentions -------------------------------------------------------------------
def test_slugs_unique_and_resolvable(ix):
    slugs = set(ix["slug_of"].values())
    assert len(slugs) == len(ROSTER)
    assert ix["slug_of"][CC] == "opus-4-5-claude-code" and ix["slug_of"]["o48"] == "claude-opus-4-8"
    assert vl.resolve_agent("claude-opus-4-8") == "o48" and vl.resolve_agent("o48") == "o48"
    assert vl.resolve_agent("nope") is None
    assert vl.family_group("Moonshot") == "other" and vl.family_group("Google") == "google"


def test_mentions_positive_and_negative():
    m = vl.build_mentions(ROSTER)
    f = lambda t, ex=None: vl.mentions(t, m, ex)  # noqa: E731
    assert f("thanks Claude Opus 4.8!") == {"o48"}
    assert f("opus 4.8 said") == {"o48"}
    assert f("Gemini 2.5 is struggling") == {"g25"} and f("@Gemini 2.5 Pro hi") == {"g25"}
    assert f("GPT-5.2 is wrong") == {"g52"}  # NOT GPT-5
    assert f("GPT-5 is right") == {"g5"}
    assert f("GPT-5.3 rumors") == set()  # no '.digit' continuation (not GPT-5)
    assert f("gpt 5.2 / gpt5.2") == {"g52"}  # separator variants
    assert f("o3 agrees") == {"o3"}
    assert f("demo3 foo3 o3x") == set()  # o3 only as a whole token
    assert f("Gemini and Claude and GPT") == set()  # bare family words never attributed
    assert f("Opus 4.5 too") == set()  # ambiguous short alias -> dropped
    assert f("Claude Opus 4.5 too") == {"o45"}  # full name is kept
    assert f("The Fine-Tuned Leader speaks") == {"ftl"}
    assert f("[Temporary] Fine-tuned Leader agrees") == {"tftl"}
    assert f("Claude Opus 4.8 on Claude Opus 4.8", ex="o48") == set()  # self-mention excluded


# ---- overview / day / event ---------------------------------------------------------------------
def test_overview_days_and_headlines(ix):
    ov = vl.overview()
    d5 = next(d for d in ov["days"] if d["day"] == 5)
    assert d5["headline"]["kind"] == "operator" and d5["headline"]["ei"] == 1
    assert d5["n_help"] == 1 and d5["n_ops"] == 1 and d5["fg"]["google"] == 2 and d5["cc"] == 1
    d6 = next(d for d in ov["days"] if d["day"] == 6)
    assert d6["headline"]["kind"] in ("talked_about", "top_speaker")  # 'automated' is not an operator
    assert ov["day_range"] == [5, 6]
    assert ov["presence"]["gemini-2-5-pro"] == {5: 2}
    assert {a["slug"] for a in ov["agents"]} >= {"claude-opus-4-8", "opus-4-5-claude-code"}


def test_day_light_events_and_hc(ix):
    d = vl.day(5)
    kinds = [(e["ei"], e["k"]) for e in d["events"]]
    assert kinds == sorted(kinds) and (100, "a") in kinds  # activity merged in ei order
    zak = next(e for e in d["events"] if e["ei"] == 1)
    assert zak["hc"] == "operator" and zak["fg"] == "human" and zak["slug"] is None and zak["p"] == ""
    cce = next(e for e in d["events"] if e["ei"] == 6)
    assert cce["p"] == "cc" and cce["slug"] == "opus-4-5-claude-code"
    assert next(e for e in d["events"] if e["ei"] == 3)["p"] == "tierb"
    help_ = next(e for e in d["events"] if e["ei"] == 101)
    assert help_["type"] == "REQUEST_HUMAN_HELPER" and "fix my VNC" in help_["s"]
    auto = next(e for e in vl.day(6)["events"] if e["ei"] == 10)
    assert auto["hc"] == "automated"
    assert d["talked_about"][0]["slug"] == "gemini-2-5-pro"
    with pytest.raises(ValueError, match="nearest active"):
        vl.day(99)


def test_day_full_text_vs_preview(ix, monkeypatch):
    """The timeline shows messages expanded, so /api/day?full=1 must ship complete text; the default
    stays a preview flagged `more` (checked the other way too: shrink the preview and watch it truncate)."""
    monkeypatch.setattr(vl, "TALK_PREVIEW", 10)
    src = "o3 and GPT-5 agree; Opus 4.5 too"
    prev = next(e for e in vl.day(5)["events"] if e["ei"] == 5)
    full = next(e for e in vl.day(5, full=True)["events"] if e["ei"] == 5)
    assert prev["c"] == src[:10] and prev["more"] is True
    assert full["c"] == src and full["more"] is False


def test_event_probe_routing(ix):
    assert vl.event(6)["probe"]["tier"] == "cc"
    assert vl.event(3)["probe"] == {"tier": "tierb", "agent": "claude-opus-4-8", "agent_id": "o48", "ei": 3}
    assert vl.event(1)["probe"] is None and vl.event(100)["probe"] is None
    assert vl.event(3)["mentions"] == ["gemini-2-5-pro"]


# ---- search ------------------------------------------------------------------------------------
def test_search_grammar_and_jumps(ix):
    assert vl.search("447")["jump"] == {"day": 447}
    assert vl.search("d5")["jump"] == {"day": 5}
    assert vl.search("2026-09-06")["jump"] == {"day": 6}
    assert vl.search("ei:3")["jump"] == {"day": 5, "ei": 3}
    assert vl.search("from:claude-opus-4-8")["total"] == 2
    assert vl.search("from:google")["total"] == 2
    assert vl.search("from:operator")["total"] == 1  # zak, not 'automated'
    assert vl.search("from:human")["total"] == 2
    ab = vl.search("about:gemini-2-5-pro")  # zak's goal post + ei 3 ('Gemini 2.5')
    assert {h["ei"] for h in ab["hits"]} == {1, 3}
    assert vl.search("room:best")["total"] == 2
    assert vl.search('"pile on" gemini')["total"] == 1
    assert vl.search("in:thinking kind")["total"] == 1  # thinking haystack ('be kind')
    assert vl.search("kind:human take")["total"] == 1
    for bad in ("", "from:nobody", "about:nobody", "day:x", "in:screens", "kind:robot"):
        with pytest.raises(ValueError):
            vl.search(bad)


def test_search_days_prefilter_equals_restricted(ix):
    full = vl.search("the")
    pre = vl.search("the", days=(5, 5))
    assert pre["total"] == full["per_day"].get(5, 0) and set(pre["per_day"]) <= {5}
    assert vl.search("the day:5")["total"] == pre["total"]
    # negative: without the prefilter the other day's hits are there
    assert full["total"] >= pre["total"]


def test_search_offset_limit(ix):
    a = vl.search("from:gemini-2-5-pro", limit=1)
    b = vl.search("from:gemini-2-5-pro", limit=1, offset=1)
    assert a["total"] == b["total"] == 2 and a["truncated"] and not b["truncated"]
    assert a["hits"][0]["ei"] != b["hits"][0]["ei"]


# ---- approximate path: anchors / actual / presence ---------------------------------------------
def test_anchor_before_excludes_after_includes(ix):
    chat = ix["chat"]
    before = tb.observable_context(3, chat, 99)
    after = tb.observable_context(3, chat, 99, include_target=True)
    assert [m["ei"] for m in before["messages"]] == [1, 2]
    assert [m["ei"] for m in after["messages"]] == [1, 2, 3]
    assert [m["ei"] for m in tb.observable_context(3, chat, 1)["messages"]] == [2]


def test_actual_rules(ix):
    chat = ix["chat"]
    assert tb.actual_for("o48", 3, "before", chat)["kind"] == "this"
    a = tb.actual_for("o48", 3, "after", chat)  # author/after: next, any gap
    assert a["kind"] == "next" and a["ei"] == 8
    other = tb.actual_for("g25", 3, "after", chat)  # other agent: next within 2h
    assert other["kind"] == "next" and other["ei"] == 7 and 0 < other["gap_s"] <= 7200
    assert tb.actual_for("o48", 7, "after", chat)["kind"] == "none"  # o48's next is >2h later
    assert tb.actual_for("g5", 3, "after", chat)["kind"] == "none"  # g5 never posts in R1 again


def test_presence_ladder(ix):
    chat, acts = ix["chat"], ix["acts"]
    assert tb.presence("g25", 3, chat, acts)["level"] == "here"
    assert tb.presence("o48", 7, chat, acts)["level"] == "here"  # its ei 3 was 35 min earlier
    assert tb.presence("g5", 3, chat, acts)["level"] == "elsewhere"  # posted only in R2 that day
    assert tb.presence("idle", 3, chat, acts)["level"] == "absent"
    assert tb.presence("o45", 3, chat, acts)["level"] == "elsewhere"  # activity-only that day


def test_probe_prompt_tierb_routes(ix):
    pp = vl.probe_prompt({"target": "tierb", "ei": 3})
    assert pp["resolved"]["anchor"] == "before" and pp["actual"]["kind"] == "this"
    assert "pick ONE lead helper" not in "\n".join(m["text"] for m in pp["messages"])
    assert pp["fidelity"]["level"] == "approximate" and pp["context"]["n"] == 2
    other = vl.probe_prompt({"target": "tierb", "ei": 2, "agent": "claude-opus-4-8", "anchor": "before"})
    assert other["resolved"]["anchor"] == "after"  # coerced: others only 'after'
    assert "I'm stuck" in "\n".join(m["text"] for m in other["messages"])
    assert any(w["code"] == "cross_anchor" for w in other["fidelity"]["warnings"])
    cf = vl.probe_prompt({"target": "tierb", "ei": 2, "agent": "claude-3-7-sonnet"})
    assert cf["fidelity"]["counterfactual"] and any(
        w["severity"] == "counterfactual" for w in cf["fidelity"]["warnings"]
    )
    assert any(w["code"] == "model_mismatch" for w in cf["fidelity"]["warnings"])
    assert not any(w["code"] == "model_mismatch" for w in pp["fidelity"]["warnings"])  # opus-4-8 voiced by itself
    with pytest.raises(ValueError, match="human"):
        vl.probe_prompt({"target": "tierb", "ei": 1})
    with pytest.raises(ValueError, match="activity"):
        vl.probe_prompt({"target": "tierb", "ei": 100})
    assert pp["sha256"] == vl.prompt_sha(pp["system"], pp["messages"])
    assert pp["format"] == "turns" and "earlier turns" in pp["fidelity"]["details"]  # the default


def test_probe_prompt_refuses_unroutable_model(ix):
    with pytest.raises(ValueError, match="OpenRouter id"):
        vl.probe_prompt({"target": "tierb", "ei": 2, "agent": "claude-opus-4-8", "model": "grok-4"})
    ok = vl.probe_prompt({"target": "tierb", "ei": 2, "agent": "claude-opus-4-8", "model": "x-ai/grok-4.5"})
    assert ok["model"] == "x-ai/grok-4.5"


def test_probe_prompt_turns_format(ix):
    tr = vl.probe_prompt({"target": "tierb", "ei": 2, "agent": "claude-opus-4-8", "format": "transcript"})
    tu = vl.probe_prompt({"target": "tierb", "ei": 2, "agent": "claude-opus-4-8"})  # turns is the default
    assert tu["format"] == "turns" and tu["sha256"] != tr["sha256"]
    assert tu["messages"][-1]["t"] == "user" and "I'm stuck" in tu["messages"][-1]["text"]
    assert "earlier turns" in tu["fidelity"]["details"] and "earlier turns" in tu["system"]
    same = lambda c: {k: v for k, v in c.items() if k != "approx_tokens"}  # noqa: E731  (the note's length differs)
    assert same(tu["context"]) == same(tr["context"]) and tu["actual"] == tr["actual"]  # same moment and comparison
    with pytest.raises(ValueError, match="format"):
        vl.probe_prompt({"target": "tierb", "ei": 2, "format": "xml"})
    assert vl.probe_prompt({"target": "cc", "at_ei": 6})["format"] is None  # the CC replay is always turns


# ---- claude code ---------------------------------------------------------------------------------
def test_cc_gap_anchor_invariant_synthetic(ix):
    turns = cc.timeline(_cc_rows())
    assert len(turns) == 3
    for a, b in zip(turns, turns[1:]):
        assert a["anchor_after"] == b["anchor_before"]


def test_cc_posting_turn_and_time_match(ix):
    link = vl.cc_link(6)
    assert link["turn"] == 1 and link["anchor_before"] == 3
    pp = vl.probe_prompt({"target": "cc", "at_ei": 6})
    assert pp["resolved"]["match"] == "posting_turn" and pp["resolved"]["turn"] == 1 and pp["resolved"]["seq"] == 3
    assert pp["actual"]["kind"] == "this" and pp["actual"]["ei"] == 6
    assert pp["fidelity"]["level"] == "near_faithful" and pp["resample_ok"] is True
    mm = [w for w in pp["fidelity"]["warnings"] if w["code"] == "model_mismatch"]
    assert mm and "claude-opus-4-8" in mm[0]["text"]  # voiced by the default model, not Opus 4.5
    era = vl.probe_prompt({"target": "cc", "at_ei": 6, "model": "claude-opus-4-5-20251101"})
    assert not any(w["code"] == "model_mismatch" for w in era["fidelity"]["warnings"])
    near = vl.probe_prompt({"target": "cc", "at_ei": 6, "model": "claude-opus-4"})  # a prefix, not the same model
    assert any(w["code"] == "model_mismatch" for w in near["fidelity"]["warnings"])
    assert "hello village" not in json.dumps(pp["messages"])  # before the posting turn
    after = vl.probe_prompt({"target": "cc", "at_ei": 6, "anchor": "after"})
    assert "hello village" in json.dumps(after["messages"]) and after["actual"]["turn"] == 2
    # someone else's message at 09:10 -> last completed turn ends 09:04 (turn 1), last read = turn 0
    m = vl.cc_at(7)
    assert m["turn"] == 1 and m["seq"] == 6 and abs(m["gap_s"] - 360) < 1 and m["last_read"]["turn"] == 0
    tp = vl.probe_prompt({"target": "cc", "at_ei": 7})
    codes = [w["code"] for w in tp["fidelity"]["warnings"]]
    assert tp["resolved"]["match"] == "time" and {"cc_time_match", "cc_unread", "cross_anchor"} <= set(codes)
    assert vl.cc_at(1) is None  # 08:00: before any CC turn
    with pytest.raises(ValueError, match="6h"):
        vl.probe_prompt({"target": "cc", "at_ei": 9})  # next day, >6h after last turn


def test_cc_window_and_resample_gate(ix):
    w = vl.cc_window(ix["segs"][0]["seg_id"])
    assert [t["chat_ei"] for t in w["turns"]] == [[], [6], []]
    seg = ix["segs"][0]["seg_id"]
    g = vl.probe_prompt({"target": "cc", "seg_id": seg, "seq": w["turns"][2]["anchor_before"], "mode": "resample"})
    assert g["resample_ok"] and g["resolved"]["turn"] == 2 and g["actual"]["turn"] == 2
    with pytest.raises(ValueError, match="resample"):  # ends on the agent's own turn
        vl.probe_prompt({"target": "cc", "seg_id": seg, "seq": 2, "mode": "resample"})


SUMMARY = "This session is being continued from a previous conversation that ran out of context. Summary: ..."


@pytest.fixture()
def ix_windows():
    """Four CC windows: the start of the logs (init at seq 0), a compaction that opens with Claude Code's
    summary, a resumed session (a later init), and a compaction whose summary row is absent."""

    def turn(seq, mm, text):
        return [_row(seq, "text", "assistant", 5, 10, mm, text=text), _row(seq + 1, "tool_result", "user", 5, 10, mm)]

    rows = (
        [_row(0, "init", "system", 5, 10, 0, text="{}"), _row(1, "user_text", "user", 5, 10, 0, text="begin")]
        + turn(2, 1, "first")
        + [_row(4, "compact", "system", 5, 10, 2), _row(5, "user_text", "user", 5, 10, 2, text=SUMMARY)]
        + turn(6, 3, "after compaction")
        + [_row(8, "init", "system", 5, 10, 4, text="{}")]
        + turn(9, 5, "after resume")
        + [_row(11, "compact", "system", 5, 10, 6)]
        + turn(12, 7, "no summary")
    )
    cc.set_day_ranges(_day_ranges(5))
    segs = cc.segments(rows)
    vl.set_index(vl._build_from([], [], NAMES, ROSTER, rows, segs, cc.day_dates()))
    yield {s["seg_id"]: s for s in segs}
    vl.set_index(None)


def _fid(seg):
    pp = vl.probe_prompt({"target": "cc", "seg_id": seg["seg_id"], "seq": seg["end_seq"]})
    w = vl.cc_window(seg["seg_id"])["summary"]
    assert w["fidelity"] == pp["fidelity"]["details"]
    return pp, w


def test_cc_fidelity_compact_vs_init_windows(ix_windows):
    segs = ix_windows
    # segmentation is unchanged: a new window at every init and compact_boundary
    assert [(s["start_seq"], s["boundary"], s["has_summary"]) for s in segs.values()] == [
        (0, "init", False),
        (4, "compact", True),
        (8, "init", False),
        (11, "compact", False),
    ]
    resumed_item = "the earlier context of this resumed session (not replayed)"
    summary_item = "the compaction summary this window began with (not in the dataset)"
    for seg in segs.values():
        pp, w = _fid(seg)
        assert "fresh" not in w["fidelity"].lower() and "NOT in the dataset" not in w["fidelity"]
    first, comp, resumed, bare = (_fid(segs[i]) for i in sorted(segs))
    # a compaction window opens with Claude Code's own summary, which is replayed: nothing claimed missing
    assert comp[0]["messages"][0]["text"].startswith("This session is being continued")
    assert summary_item not in comp[0]["fidelity"]["missing"] and resumed_item not in comp[0]["fidelity"]["missing"]
    assert "summary" in comp[1]["fidelity"] and "replayed" in comp[1]["fidelity"]
    # a resumed session: the earlier context the agent kept is listed as missing
    assert resumed_item in resumed[0]["fidelity"]["missing"] and resumed[1]["resumed"] is True
    assert "resumed" in resumed[1]["fidelity"] and "after resume" in json.dumps(resumed[0]["messages"])
    assert "after compaction" not in json.dumps(resumed[0]["messages"])  # the replay starts at the resume
    # the very start of the logs has no earlier context
    assert resumed_item not in first[0]["fidelity"]["missing"] and first[1]["resumed"] is False
    assert "start of its Claude Code logs" in first[1]["fidelity"]
    # a compaction whose summary row is absent says so
    assert summary_item in bare[0]["fidelity"]["missing"]


# ---- probe jobs / log ------------------------------------------------------------------------------
def test_probe_jobs_roundtrip_and_followup(ix):
    calls = []

    def fake_call(system, messages, model):
        calls.append(messages)
        return {"thinking": "t", "text": f"answer {len(calls)}", "tool_use": None, "usage": {"in": 1, "out": 1}}

    with tempfile.TemporaryDirectory() as d:
        store = pj.Store(Path(d))
        jobs = pj.Jobs(store, vl.probe_prompt, fake_call, workers=1)
        j = jobs.submit({"target": "tierb", "ei": 3, "question": "why?"})
        jobs.pool.shutdown(wait=True)
        rec = store.get(j["probe_id"])
        pp = vl.probe_prompt({"target": "tierb", "ei": 3, "question": "why?"})
        assert rec["status"] == "done" and rec["text"] == "answer 1"
        assert rec["sha256"] == pp["sha256"] == vl.prompt_sha(rec["system"], rec["messages"])
        jobs.pool = pj.ThreadPoolExecutor(max_workers=1)
        f = jobs.submit({"parent_id": j["probe_id"], "question": "and then?"})
        jobs.pool.shutdown(wait=True)
        fr = store.get(f["probe_id"])
        assert fr["messages"][-2:] == [{"t": "assistant", "text": "answer 1"}, {"t": "user", "text": "and then?"}]
        assert fr["messages"][:-2] == rec["messages"] and fr["root_id"] == j["probe_id"]
        assert fr["resolved"] == rec["resolved"] and fr["model"] == rec["model"]
        assert rec["format"] == fr["format"] == "turns"
        assert fr["sha256"] == vl.prompt_sha(fr["system"], fr["messages"]) != rec["sha256"]
        assert calls[-1] == fr["messages"]  # what was sent == what was logged
        # persisted + reloadable; counts only finished ROOT probes
        store2 = pj.Store(Path(d))
        assert set(store2.probes) == {j["probe_id"], f["probe_id"]}
        assert store2.n_probes_by_ei() == {3: 1} and store2.probes_by_day() == {5: 1}
        with pytest.raises(ValueError):
            jobs.submit({"parent_id": "nope", "question": "x"})
        with pytest.raises(ValueError):
            jobs.submit({"target": "tierb"})  # bad params fail fast
        # an error is recorded, not raised
        jobs.call_fn = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
        jobs.pool = pj.ThreadPoolExecutor(max_workers=1)
        e = jobs.submit({"target": "tierb", "ei": 3})
        jobs.pool.shutdown(wait=True)
        er = store.get(e["probe_id"])
        assert er["status"] == "error" and "boom" in er["error"]
        assert jobs.get(e["job_id"])["status"] == "error"
        # a non-default format is recorded and inherited by follow-ups
        jobs.call_fn = fake_call
        jobs.pool = pj.ThreadPoolExecutor(max_workers=1)
        t = jobs.submit({"target": "tierb", "ei": 2, "agent": "claude-opus-4-8", "format": "transcript"})
        jobs.pool.shutdown(wait=True)
        jobs.pool = pj.ThreadPoolExecutor(max_workers=1)
        tf = jobs.submit({"parent_id": t["probe_id"], "question": "more?"})
        jobs.pool.shutdown(wait=True)
        assert store.get(t["probe_id"])["format"] == store.get(tf["probe_id"])["format"] == "transcript"


def test_stars_persist(ix):
    with tempfile.TemporaryDirectory() as d:
        s = pj.Store(Path(d))
        a = s.star_add(ei=3, day=5, note="look")
        assert s.star_add(ei=3, day=5)["id"] == a["id"]  # idempotent
        assert s.starred_eis() == {3} and s.stars_by_day() == {5: 1}
        assert pj.Store(Path(d)).starred_eis() == {3}
        assert s.star_remove(ei=3) == 1 and pj.Store(Path(d)).starred_eis() == set()
        with pytest.raises(ValueError):
            s.star_remove(ei=3)


# ---- real data (opt-in) ------------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("VILLAGE_DATA_TESTS") != "1", reason="set VILLAGE_DATA_TESTS=1 (loads the real CC corpus)"
)
def test_real_cc_gap_anchors():
    """turns[i].anchor_after == turns[i+1].anchor_before across all segments, except where control or
    user rows sit between turns (listed, and bounded)."""
    rows = cc.load_all()
    bad, total = [], 0
    for s in cc.segments(rows):
        tl = cc.timeline(cc.segment_slice(rows, s))
        for a, b in zip(tl, tl[1:]):
            total += 1
            if a["anchor_after"] != b["anchor_before"]:
                gap = rows[a["anchor_after"] + 1 : b["anchor_before"] + 1]
                assert all(r.role != "assistant" for r in gap)  # only non-assistant rows in the gap
                bad.append((s["seg_id"], a["turn"]))
    print(f"\n{len(bad)} / {total} turn pairs have a gap between anchor_after and anchor_before; first: {bad[:5]}")
    assert len(bad) / max(total, 1) < 0.05

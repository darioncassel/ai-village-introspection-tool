"""Unit tests for tierb_lib pure functions (no dataset, no API)."""

import pytest

from village_introspect import tierb_lib as tb
from village_introspect.cc_lib import _BELIEF_PREFIX

_PAUSE = _BELIEF_PREFIX.strip()[:30]  # "(A pause before your next step"


def _cr(ei, room, sid, speaker, family, is_agent, content, day, thinking="", kind=""):
    return {
        "ei": ei,
        "room": room,
        "sid": sid,
        "speaker": speaker,
        "family": family,
        "is_agent": is_agent,
        "content": content,
        "day": day,
        "created_at": f"t{ei}",
        "thinking": thinking,
        "think_kind": kind,
    }


def test_is_cc_prefix():
    assert tb.is_cc("claude-code::claude-opus-4-5-20251101")
    assert not tb.is_cc("claude-opus-4-5-20251101")  # the OTHER Opus 4.5 (regular scaffold)
    assert not tb.is_cc("gpt-5.5") and not tb.is_cc("") and not tb.is_cc(None)


def test_classify_family():
    assert tb.classify_family("Claude Opus 4.8") == "Anthropic"
    assert tb.classify_family("Gemini 2.5 Pro") == "Google"
    assert tb.classify_family("GPT-5.5") == "OpenAI"
    assert tb.classify_family("o3") == "OpenAI"
    assert tb.classify_family("DeepSeek-V3.2") == "DeepSeek"
    assert tb.classify_family("Kimi K2.6") == "Moonshot"
    assert tb.classify_family("Grok 4") == "xAI"


def test_extract_thinking_anthropic():
    out = {"content": [{"type": "thinking", "thinking": "secret plan"}, {"type": "text", "text": "hi"}]}
    assert tb.extract_thinking(out) == ("secret plan", "verbatim")


def test_extract_thinking_gemini():
    out = {"candidates": [{"content": {"parts": [{"thought": True, "text": "gthink"}, {"text": "answer"}]}}]}
    assert tb.extract_thinking(out) == ("gthink", "verbatim")


def test_extract_thinking_openai_summary_only():
    out = [{"type": "reasoning", "summary": [{"text": "s1"}, {"text": "s2"}]}, {"type": "message"}]
    assert tb.extract_thinking(out) == ("s1\ns2", "summary")


def test_extract_thinking_anthropic_bare_block_list():
    out = [{"type": "thinking", "thinking": "plan A", "signature": "x"}, {"type": "text", "text": "hi"}]
    assert tb.extract_thinking(out) == ("plan A", "verbatim")
    # empty thinking (display omitted) and redacted blocks carry no text
    assert tb.extract_thinking([{"type": "thinking", "thinking": ""}, {"type": "redacted_thinking"}]) == ("", "")


def test_extract_thinking_chat_completions_reasoning():
    assert tb.extract_thinking({"role": "assistant", "content": "hi", "reasoning": "plan B"}) == ("plan B", "reasoning")
    out = {"role": "assistant", "content": "", "reasoning_content": "plan C", "tool_calls": []}
    assert tb.extract_thinking(out) == ("plan C", "reasoning")
    assert tb.extract_thinking({"role": "assistant", "content": "x", "reasoning": None}) == ("", "")
    assert tb.extract_thinking({"role": "assistant", "content": "x", "reasoning_content": " "}) == ("", "")


def test_extract_thinking_none():
    assert tb.extract_thinking(None) == ("", "")
    assert tb.extract_thinking({"role": "assistant", "content": "x"}) == ("", "")  # chat-completion shape


def test_observable_context_window_and_room():
    A, B = "roomA", "roomB"
    chat = [
        _cr(1, A, "u", "op", "human", False, "goal: help", 5),
        _cr(2, A, "g", "Gemini", "Google", True, "m2", 5),
        _cr(3, B, "x", "GPT", "OpenAI", True, "other room", 5),  # different room -> excluded
        _cr(4, A, "h", "Haiku", "Anthropic", True, "m4", 5),
        _cr(5, A, "o", "Opus", "Anthropic", True, "m5", 5),
        _cr(6, A, "o", "Opus", "Anthropic", True, "TARGET", 5, thinking="th"),
    ]
    oc = tb.observable_context(6, chat, window=2)
    assert oc["room"] == A
    assert [m["ei"] for m in oc["messages"]] == [4, 5]  # last 2 in roomA before ei 6
    assert oc["target"]["content"] == "TARGET" and oc["target"]["thinking"] == "th"
    # full window keeps room filter (excludes roomB) and order
    oc2 = tb.observable_context(6, chat, window=99)
    assert [m["ei"] for m in oc2["messages"]] == [1, 2, 4, 5]


def _all_text(pp):
    return "\n".join(m["text"] for m in pp["messages"])


def test_build_probe_prompt_belief_and_resample(monkeypatch):
    monkeypatch.setattr(
        tb,
        "_ROSTER",
        {
            "g": {
                "agent_id": "g",
                "name": "Gemini 2.5 Pro",
                "model_string": "gemini-2.5-pro",
                "family": "Google",
                "tier": "tierb",
            }
        },
    )
    chat = [
        _cr(1, "A", "u", "operator", "human", False, "Help Gemini!", 447),
        _cr(2, "A", "o", "Claude Opus 4.8", "Anthropic", True, "there is no adversary", 447),
        _cr(3, "A", "g", "Gemini 2.5 Pro", "Google", True, "TARGET", 447, thinking="I feel blocked"),
    ]
    pp = tb.build_probe_prompt("g", 3, "", mode="belief", window=99, chat=chat)
    assert pp["messages"][-1]["t"] == "user" and pp["n_context"] == 2
    txt = pp["messages"][-1]["text"]
    assert "Help Gemini!" in txt and "there is no adversary" in txt  # observable context present
    assert "TARGET" not in txt  # target itself not in context
    assert "what will you do next" in txt.lower()
    pp2 = tb.build_probe_prompt("g", 3, "", mode="resample", window=99, chat=chat)
    assert "next message to the room" in pp2["messages"][-1]["text"].lower()
    assert _PAUSE not in _all_text(pp2)  # a resample is not a private aside
    assert "RECONSTRUCTION NOTE" in pp["system"] and "Gemini 2.5 Pro" in pp["system"]
    assert pp["format"] == "turns" and "your earlier turns" in pp["system"]  # the default
    tr = tb.build_probe_prompt("g", 3, "", window=99, chat=chat, fmt="transcript")
    assert tr["format"] == "transcript" and "as a transcript" in tr["system"] and len(tr["messages"]) == 1
    assert "Help Gemini!" in tr["messages"][0]["text"] and "TARGET" not in tr["messages"][0]["text"]


def test_build_probe_prompt_anchor_after_and_other_agent_coercion(monkeypatch):
    _roster_go(monkeypatch)
    chat = [
        _cr(1, "A", "u", "operator", "human", False, "Help Gemini!", 447),
        _cr(2, "A", "g", "Gemini 2.5 Pro", "Google", True, "I'm stuck", 447),
        _cr(3, "A", "o", "Claude Opus 4.8", "Anthropic", True, "one lead helper", 447),
    ]
    for r in chat:  # parseable times: 'next within 2h' needs real gaps
        r["created_at"] = f"2026-06-22 17:0{r['ei']}:00"
    before = tb.build_probe_prompt("g", 2, "", window=99, chat=chat)
    after = tb.build_probe_prompt("g", 2, "", window=99, chat=chat, anchor="after")
    assert before["anchor"] == "before" and "I'm stuck" not in _all_text(before)
    assert after["anchor"] == "after" and "I'm stuck" in _all_text(after)
    assert before["actual"]["ei"] == 2 and after["actual"] is None  # g never posts again
    # asking ANOTHER agent at Gemini's message is always 'after' (it could only react once it saw it)
    other = tb.build_probe_prompt("o", 2, "", window=99, chat=chat, anchor="before")
    assert other["anchor"] == "after" and "I'm stuck" in _all_text(other)
    assert other["actual"]["ei"] == 3 and other["actual_relation"] == "its next message in this room"
    try:
        tb.build_probe_prompt("g", 2, "", chat=chat, anchor="sideways")
        raise AssertionError("bad anchor accepted")
    except ValueError:
        pass


def _roster_go(monkeypatch):
    ros = {
        "g": {"agent_id": "g", "name": "Gemini 2.5 Pro", "model_string": "gemini-2.5-pro", "family": "Google"},
        "o": {"agent_id": "o", "name": "Claude Opus 4.8", "model_string": "claude-opus-4-8", "family": "Anthropic"},
        "k": {"agent_id": "k", "name": "Grok 4.5", "model_string": "grok-4.5", "family": "xAI"},
        "f": {"agent_id": "f", "name": "GPT-5.5", "model_string": "gpt-5.5", "family": "OpenAI"},
        "p": {"agent_id": "p", "name": "Gemini 3.1 Pro", "model_string": "gemini-3.1-pro-preview", "family": "Google"},
        "h": {"agent_id": "h", "name": "Claude Haiku 4.5", "model_string": "claude-haiku-4-5-20251001"},
        "cc": {
            "agent_id": "cc",
            "name": "Opus 4.5 (Claude Code)",
            "model_string": "claude-code::claude-opus-4-5-20251101",
        },
        "x": {"agent_id": "x", "name": "GPT-5.4", "model_string": "gpt-5.4-2026-03-05", "family": "OpenAI"},
    }
    monkeypatch.setattr(tb, "_ROSTER", ros)


def test_turns_format_own_messages_become_assistant_turns(monkeypatch):
    _roster_go(monkeypatch)
    chat = [
        _cr(1, "A", "u", "operator", "human", False, "Help Gemini!", 447),
        _cr(2, "A", "o", "Claude Opus 4.8", "Anthropic", True, "glad to help", 447),
        _cr(3, "A", "o", "Claude Opus 4.8", "Anthropic", True, "one lead helper", 447),
        _cr(4, "A", "g", "Gemini 2.5 Pro", "Google", True, "I'm stuck", 447),
        _cr(5, "A", "h", "Haiku", "Anthropic", True, "me too", 447),
    ]
    pp = tb.build_probe_prompt("o", 4, "What now?", window=99, chat=chat, fmt="turns")
    ts = [m["t"] for m in pp["messages"]]
    assert ts == ["user", "assistant", "user"]  # op | its two posts (coalesced) | Gemini + the question
    assert pp["messages"][1]["text"] == "glad to help\n\none lead helper"  # its own words, no speaker label
    assert pp["messages"][0]["text"] == "operator [human]: Help Gemini!"
    last = pp["messages"][2]["text"]
    assert last.startswith("Gemini 2.5 Pro [Google]: I'm stuck") and last.endswith("What now?")
    assert "me too" not in last  # anchored right after Gemini's message
    assert "your earlier turns" in pp["system"] and "as a transcript" not in pp["system"]
    assert pp["format"] == "turns"


def test_turns_format_edges(monkeypatch):
    _roster_go(monkeypatch)
    chat = [
        _cr(1, "A", "g", "Gemini 2.5 Pro", "Google", True, "first", 447),
        _cr(2, "A", "g", "Gemini 2.5 Pro", "Google", True, "  ", 447),  # empty own message: skipped
        _cr(3, "A", "g", "Gemini 2.5 Pro", "Google", True, "TARGET", 447),
    ]
    # the author asked right after its own message: starts and ends on its own turns
    pp = tb.build_probe_prompt("g", 3, "", window=99, chat=chat, anchor="after", fmt="turns")
    ts = [m["t"] for m in pp["messages"]]
    assert ts == ["user", "assistant", "user"]  # user-first placeholder, its posts, then the question
    assert pp["messages"][0]["text"] == "(This is the start of the room's chat.)"  # nothing earlier was cut
    assert pp["messages"][1]["text"] == "first\n\nTARGET"
    assert "what will you do next" in pp["messages"][2]["text"].lower()
    assert all(m["text"].strip() for m in pp["messages"])
    for i in range(1, len(ts)):
        assert ts[i] != ts[i - 1]  # strictly alternating
    try:
        tb.build_probe_prompt("g", 3, "", window=99, chat=chat, fmt="xml")
        raise AssertionError("bad format accepted")
    except ValueError:
        pass


def _timed(chat):
    for r in chat:  # parseable times, a minute apart
        r["created_at"] = f"2026-06-22 17:{r['ei']:02d}:00.5"
    return chat


def test_resample_ends_on_in_frame_cue_without_belief_prefix(monkeypatch):
    _roster_go(monkeypatch)
    assert tb._RESAMPLE_Q.startswith("(Your turn:")
    chat = [
        _cr(1, "A", "u", "operator", "human", False, "Help Gemini!", 447),
        _cr(2, "A", "o", "Claude Opus 4.8", "Anthropic", True, "one lead helper", 447),
        _cr(3, "A", "g", "Gemini 2.5 Pro", "Google", True, "I'm stuck", 447),
    ]
    for fmt in ("turns", "transcript"):
        # the context ends on another speaker's (user) turn: the cue folds into it
        pp = tb.build_probe_prompt("o", 3, "ignored in resample", mode="resample", window=99, chat=chat, fmt=fmt)
        last = pp["messages"][-1]
        assert last["t"] == "user" and last["text"].endswith("\n\n" + tb._RESAMPLE_Q)
        assert last["text"].rsplit("\n\n", 1)[1].startswith("(Your turn:")
        assert _PAUSE not in _all_text(pp) and "ignored in resample" not in _all_text(pp)
        assert "I'm stuck" in last["text"]
        belief = tb.build_probe_prompt("o", 3, "", window=99, chat=chat, fmt=fmt)
        assert _PAUSE in _all_text(belief) and len(belief["messages"]) == len(pp["messages"])
    # the context ends on the agent's own (assistant) turn: the cue is a user turn of its own
    pp = tb.build_probe_prompt("o", 2, "", mode="resample", window=99, chat=chat, anchor="after", fmt="turns")
    assert [m["t"] for m in pp["messages"]] == ["user", "assistant", "user"]
    assert pp["messages"][-1]["text"] == tb._RESAMPLE_Q
    # no context at all (the author asked just before the room's first message)
    first = [_cr(1, "B", "g", "Gemini 2.5 Pro", "Google", True, "first ever", 447)]
    for fmt in ("turns", "transcript"):
        pp = tb.build_probe_prompt("g", 1, "", mode="resample", window=99, chat=first, fmt=fmt)
        assert pp["messages"] == [{"t": "user", "text": tb._RESAMPLE_Q}]


def test_turns_opener_says_earlier_messages_are_missing_only_when_cut(monkeypatch):
    _roster_go(monkeypatch)
    chat = [
        _cr(1, "A", "u", "operator", "human", False, "hello", 447),
        _cr(2, "A", "g", "Gemini 2.5 Pro", "Google", True, "first", 447),
        _cr(3, "A", "g", "Gemini 2.5 Pro", "Google", True, "second", 447),
    ]
    assert tb.observable_context(3, chat, window=99)["truncated"] is False
    assert tb.observable_context(3, chat, window=2)["truncated"] is False  # exactly the room's 2 earlier ones
    assert tb.observable_context(3, chat, window=1)["truncated"] is True
    cut = tb.build_probe_prompt("g", 3, "", window=1, chat=chat, fmt="turns")  # context = [its ei 2]; ei 1 cut
    assert cut["messages"][0]["text"] == "(Earlier messages in this room are not included.)"
    assert "only the recent chat" in cut["system"]
    whole = tb.build_probe_prompt("g", 3, "", window=99, chat=chat, fmt="turns")  # starts on the operator
    assert whole["messages"][0]["text"].startswith("operator [human]: hello")
    assert "from its start" in whole["system"] and "only the recent chat" not in whole["system"]


def test_fidelity_and_system_describe_the_real_context_size(monkeypatch):
    _roster_go(monkeypatch)
    chat = _timed(
        [
            _cr(1, "A", "g", "Gemini 2.5 Pro", "Google", True, "the room's first message", 447),
            _cr(2, "A", "o", "Claude Opus 4.8", "Anthropic", True, "reply", 447),
            _cr(3, "A", "g", "Gemini 2.5 Pro", "Google", True, "third", 447),
        ]
    )
    empty = tb.fidelity("g", 1, "before", 80, "gemini-2.5-pro", n_context=0, chat=chat, acts=[])
    assert "80" not in empty["summary"] + empty["details"] and "no room chat" in empty["details"]
    part = tb.fidelity("g", 3, "before", 80, "gemini-2.5-pro", n_context=2, chat=chat, acts=[])
    assert "most recent" not in part["details"] and "80" not in part["details"]
    assert "whole chat" in part["details"] and "(2 chat messages)" in part["details"]
    assert "earlier turns" in part["details"]
    one = tb.fidelity("o", 2, "before", 80, "claude-opus-4-8", n_context=1, chat=chat, acts=[], fmt="transcript")
    assert "(1 chat message)" in one["summary"] and "as a transcript" in one["details"]
    full = tb.fidelity("g", 3, "before", 2, "gemini-2.5-pro", n_context=2, chat=chat, acts=[])
    assert "the most recent 2 chat messages" in full["details"] and full["summary"].startswith(
        "Rebuilt from the last 2"
    )
    # the prompt itself: no context -> no leading blank lines, and the system note says there is no chat
    for fmt in ("transcript", "turns"):
        pp = tb.build_probe_prompt("g", 1, "", window=80, chat=chat, fmt=fmt)
        assert pp["n_context"] == 0 and pp["messages"][0]["text"].startswith("(A pause")
        assert "no chat follows" in pp["system"] and "What follows is" not in pp["system"]


def test_same_model_is_exact_after_normalising():
    same = [
        ("claude-opus-4-8", "claude-opus-4-8"),
        ("google/gemini-3.1-pro-preview", "gemini-3.1-pro-preview"),
        ("claude-opus-4-5-20251101", "claude-code::claude-opus-4-5-20251101"),
        ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
        ("gpt-5.4", "gpt-5.4-2026-03-05"),
        ("x-ai/grok-4.5", "grok-4.5"),
        ("deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro"),
        ("GPT-5.5", "gpt-5.5"),
    ]
    differ = [
        ("gpt-5", "gpt-5.5"),
        ("openai/gpt-5", "gpt-5.1-2025-11-13"),
        ("x-ai/grok-4", "grok-4.5"),
        ("gemini-3", "gemini-3.1-pro-preview"),
        ("z-ai/glm-5", "z-ai/glm-5.2"),
        ("gpt-4", "gpt-4o-2024-08-06"),
        ("kimi-k", "kimi-k3"),
        ("claude-opus-4-8", ""),
        ("", ""),
    ]
    for a, b in same:
        assert tb.same_model(a, b), (a, b)
    for a, b in differ:
        assert not tb.same_model(a, b), (a, b)


def test_fidelity_model_mismatch_uses_exact_ids(monkeypatch):
    _roster_go(monkeypatch)
    chat = _timed(
        [_cr(i, "A", sid, sid, "?", True, f"m{i}", 447) for i, sid in enumerate(["f", "k", "o", "p", "cc", "x"], 1)]
    )

    def warns(agent, ei, model):
        f = tb.fidelity(agent, ei, "before", 80, model, n_context=ei - 1, chat=chat, acts=[])
        return any(w["code"] == "model_mismatch" for w in f["warnings"])

    assert warns("f", 1, "gpt-5") and warns("k", 2, "x-ai/grok-4")  # a different model: flagged
    assert not warns("o", 3, "claude-opus-4-8") and not warns("p", 4, "google/gemini-3.1-pro-preview")
    assert not warns("cc", 5, "claude-opus-4-5-20251101") and not warns("x", 6, "gpt-5.4")


def test_automated_nudger_is_not_labelled_human(monkeypatch):
    _roster_go(monkeypatch)
    chat = _timed(
        [
            _cr(1, "A", None, "automated", "human", False, "time to start the day", 447),
            _cr(2, "A", None, "test-visitor", "human", False, "hello room", 447),
            _cr(3, "A", "o", "Claude Opus 4.8", "Anthropic", True, "morning", 447),
        ]
    )
    tr = tb.build_probe_prompt("o", 3, "", window=99, chat=chat, fmt="transcript")["messages"][0]["text"]
    assert "automated [automated]: time to start the day" in tr and "test-visitor [human]: hello room" in tr
    tu = tb.build_probe_prompt("o", 3, "", window=99, chat=chat, fmt="turns")["messages"][0]["text"]
    assert tu.startswith("automated [automated]: time to") and "[human]: time to" not in tu
    f = tb.fidelity("o", 1, "after", 80, "claude-opus-4-8", n_context=1, chat=chat, acts=[])
    xa = [w["text"] for w in f["warnings"] if w["code"] == "cross_anchor"]
    assert xa == ["Asking Claude Opus 4.8 right after an automated message"]
    f2 = tb.fidelity("o", 2, "after", 80, "claude-opus-4-8", n_context=2, chat=chat, acts=[])
    assert [w["text"] for w in f2["warnings"] if w["code"] == "cross_anchor"] == [
        "Asking Claude Opus 4.8 right after test-visitor (human)'s message"
    ]


def test_dead_helpers_are_gone():
    assert not hasattr(tb, "agent_activity") and not hasattr(tb, "turns")


@pytest.mark.parametrize(
    "ts, micro",
    [
        ("2025-05-22 19:12:28.57899", 578990),  # 5 digits (trailing zero trimmed)
        ("2025-09-18 17:37:09.6125", 612500),
        ("2025-04-15 18:50:06.731", 731000),
        ("2025-04-15 18:50:06.73", 730000),
        ("2025-12-30 20:17:49.1", 100000),
        ("2026-09-03 21:55:03.311588", 311588),
        ("2026-09-03 21:55:03", 0),
        ("2026-09-03T21:55:03.5Z", 500000),
        ("2026-09-03 21:55:03.12345+00", 123450),
        ("2026-09-03T21:55:03.1+00:00", 100000),
        ("2026-09-03 21:55:03.1234567", 123456),  # more than 6 digits: truncated
    ],
)
def test_parse_ts_every_real_format(ts, micro):
    d = tb.parse_ts(ts)
    assert d is not None and d.utcoffset().total_seconds() == 0 and d.microsecond == micro
    assert (d.hour, d.second) in ((19, 28), (17, 9), (18, 6), (20, 49), (21, 3))


def test_parse_ts_offsets_and_garbage():
    assert tb.gap_seconds("2025-05-22 19:12:28.57899", "2025-05-22 19:13:28.578990") == 60.0
    assert tb.gap_seconds("2026-01-01 00:00:00.5", "2026-01-01T01:00:00.5+01:00") == 0.0
    assert tb.gap_seconds("2026-01-01 00:00:00", "2026-01-01 05:30:00+0530") == 0.0
    for bad in ("", None, "t3", "yesterday", "2026-13-01 00:00:00", "2026-01-01 00:00:00.5 junk"):
        assert tb.parse_ts(bad) is None, bad

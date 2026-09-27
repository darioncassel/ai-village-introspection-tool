"""Unit tests for cc_lib reconstruction (pure functions; no API calls, no dataset needed)."""

from village_introspect import cc_lib as cc


def _rows(*specs):
    """Build a CCRow list from (kind, role, text_or_toolname, [tool_input]) tuples, seq auto."""
    out = []
    for i, s in enumerate(specs):
        kind, role = s[0], s[1]
        r = cc.CCRow(seq=i, kind=kind, role=role, created_at=f"2026-01-01 00:00:{i:02d}.000000")
        if kind == "tool_use":
            r.tool_name = s[2]
            r.tool_input = s[3] if len(s) > 3 else {}
        else:
            r.text = s[2] if len(s) > 2 else ""
        out.append(r)
    return out


# ---- _classify (dataset row -> CCRow payloads) ----
def test_classify_assistant_blocks():
    row = {
        "message_type": "assistant",
        "created_at": "t",
        "content": {
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "hello"},
                    {"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}},
                ],
            }
        },
    }
    out = cc._classify(row)
    kinds = [p["kind"] for p in out]
    assert kinds == ["thinking", "text", "tool_use"]
    assert out[0]["text"] == "hmm"
    assert out[2]["tool_name"] == "Bash" and out[2]["tool_input"] == {"cmd": "ls"}
    assert all(p["msg_id"] == "msg_1" for p in out)


def test_classify_user_tool_result_list_and_str():
    row_list = {
        "message_type": "user",
        "created_at": "t",
        "content": {
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "x", "content": [{"type": "text", "text": "R1"}]}],
            }
        },
    }
    row_str = {
        "message_type": "user",
        "created_at": "t",
        "content": {
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "R2"}]}
        },
    }
    assert cc._classify(row_list)[0]["text"] == "R1"
    assert cc._classify(row_str)[0]["text"] == "R2"


def test_classify_init_captures_metadata():
    row = {
        "message_type": "system",
        "created_at": "t",
        "content": {
            "type": "system",
            "subtype": "init",
            "model": "claude-opus-4-5-20251101",
            "cwd": "/ws",
            "claude_code_version": "2.1.38",
            "permissionMode": "bypassPermissions",
            "tools": ["Bash", "Read"],
        },
    }
    p = cc._classify(row)[0]
    assert p["kind"] == "init"
    import json

    meta = json.loads(p["text"])
    assert meta["model"] == "claude-opus-4-5-20251101" and meta["tools"] == ["Bash", "Read"]


# ---- build_messages ----
def test_build_messages_coalesce_drop_thinking_render_tools():
    rows = _rows(
        ("user_text", "user", "GO: publish today's stories"),
        ("thinking", "assistant", "secret plan"),
        ("text", "assistant", "On it."),
        ("tool_use", "assistant", "Bash", {"cmd": "ls"}),
        ("tool_result", "user", "file1\nfile2"),
        ("text", "assistant", "Done."),
    )
    msgs = cc.build_messages(rows, upto_seq=5)
    # user-first, strict alternation
    assert [m["t"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    # thinking is NOT in the replayed context
    assert "secret plan" not in "\n".join(m["text"] for m in msgs)
    # assistant turn coalesces text + rendered tool call
    assert "On it." in msgs[1]["text"] and "⏺ Bash(" in msgs[1]["text"] and '"cmd": "ls"' in msgs[1]["text"]
    # observation rendered
    assert "⎿ file1\nfile2" in msgs[2]["text"]


def test_build_messages_user_first_prepend():
    rows = _rows(("text", "assistant", "starting mid-session"))
    msgs = cc.build_messages(rows, upto_seq=0)
    assert msgs[0]["t"] == "user"  # synthetic user turn prepended
    assert msgs[1]["t"] == "assistant"


def test_build_messages_respects_upto_and_no_truncation():
    big = "X" * 50000
    rows = _rows(
        ("user_text", "user", "go"),
        ("text", "assistant", "a"),
        ("tool_use", "assistant", "Read", {"path": "/big"}),
        ("tool_result", "user", big),
        ("text", "assistant", "later turn — should be excluded"),
    )
    msgs = cc.build_messages(rows, upto_seq=3)  # exclude seq 4
    joined = "\n".join(m["text"] for m in msgs)
    assert "later turn" not in joined
    assert big in joined  # observation preserved in full (no truncation)


# ---- timeline ----
def test_timeline_segments_turns_and_anchors():
    rows = _rows(
        ("user_text", "user", "kickoff"),  # seq 0
        ("thinking", "assistant", "t1"),  # seq 1  turn 0 start
        ("text", "assistant", "act one"),  # seq 2
        ("tool_use", "assistant", "Bash", {"c": 1}),  # seq 3
        ("tool_result", "user", "ok"),  # seq 4  turn 0 obs
        ("thinking", "assistant", "t2"),  # seq 5  turn 1 start
        ("tool_use", "assistant", "Read", {"p": 2}),  # seq 6
        ("tool_result", "user", "data"),  # seq 7  turn 1 obs
    )
    tl = cc.timeline(rows)
    assert len(tl) == 2
    t0 = tl[0]
    assert t0["first_seq"] == 1 and t0["last_seq"] == 4
    assert t0["anchor_before"] == 0 and t0["anchor_after"] == 4
    assert t0["thinking"] == "t1" and t0["text"] == "act one"
    assert t0["actions"] == [{"name": "Bash", "input": {"c": 1}}]
    assert t0["observations"] == ["ok"]
    assert tl[1]["first_seq"] == 5 and tl[1]["anchor_before"] == 4


# ---- build_probe_prompt ----
def test_probe_prompt_belief_appends_question():
    rows = _rows(("user_text", "user", "go"), ("text", "assistant", "hi"), ("tool_result", "user", "obs"))
    pp = cc.build_probe_prompt(rows, upto_seq=2, question="Why did you do that?", mode="belief")
    assert pp["messages"][-1]["t"] == "user"
    assert "Why did you do that?" in pp["messages"][-1]["text"]
    assert "outside the village" in pp["messages"][-1]["text"].lower()
    assert "sent to anyone" not in pp["messages"][-1]["text"]  # it IS read (logged); don't claim otherwise


def test_probe_prompt_belief_default_question():
    rows = _rows(("user_text", "user", "go"), ("text", "assistant", "hi"))
    pp = cc.build_probe_prompt(rows, upto_seq=1, question="", mode="belief")
    assert "what will you do next" in pp["messages"][-1]["text"].lower()


def test_probe_prompt_resample_requires_user_last():
    # history ends on an assistant turn -> resample must refuse
    rows = _rows(("user_text", "user", "go"), ("text", "assistant", "hi"))
    try:
        cc.build_probe_prompt(rows, upto_seq=1, question="", mode="resample")
        assert False, "expected ValueError"
    except ValueError:
        pass
    # history ends on an observation -> resample ok
    rows2 = _rows(("user_text", "user", "go"), ("text", "assistant", "hi"), ("tool_result", "user", "obs"))
    pp = cc.build_probe_prompt(rows2, upto_seq=2, question="", mode="resample")
    assert pp["messages"][-1]["t"] == "user" and pp["mode"] == "resample"


def test_segments_split_on_init_and_compact():
    rows = _rows(
        ("init", "system", '{"model":"m"}'),  # seq 0  -> segment start (init)
        ("user_text", "user", "kick"),  # seq 1
        ("text", "assistant", "a"),  # seq 2  turn
        ("tool_result", "user", "o"),  # seq 3
        ("compact", "system", "[compaction]"),  # seq 4  -> new segment (compact)
        ("thinking", "assistant", "t"),  # seq 5  turn
        ("tool_use", "assistant", "Bash", {"c": 1}),  # seq 6
        ("tool_result", "user", "o2"),  # seq 7
    )
    segs = cc.segments(rows)
    assert len(segs) == 2
    assert segs[0]["boundary"] == "init" and segs[0]["start_seq"] == 0 and segs[0]["end_seq"] == 3
    assert segs[1]["boundary"] == "compact" and segs[1]["start_seq"] == 4 and segs[1]["end_seq"] == 7
    assert not segs[0]["has_summary"] and not segs[1]["has_summary"]  # no summary row after this compaction
    # a slice reconstructs only within its window, and anchors (global .seq) stay valid on the slice
    sl = cc.segment_slice(rows, segs[1])
    tl = cc.timeline(sl)
    assert len(tl) == 1 and tl[0]["first_seq"] == 5
    msgs = cc.build_messages(sl, tl[0]["anchor_after"])
    assert "⏺ Bash(" in "\n".join(m["text"] for m in msgs)
    assert "kick" not in "\n".join(m["text"] for m in msgs)  # prior segment excluded


def test_segments_flag_the_compaction_summary():
    """Claude Code logs its compaction summary as the user message right after the boundary; the window
    replays it as its first message."""
    rows = _rows(
        ("init", "system", "{}"),
        ("text", "assistant", "a"),
        ("compact", "system", "[compaction]"),
        ("status", "system", ""),
        ("user_text", "user", "This session is being continued from a previous conversation..."),
        ("text", "assistant", "b"),
    )
    segs = cc.segments(rows)
    assert [(s["boundary"], s["has_summary"]) for s in segs] == [("init", False), ("compact", True)]
    msgs = cc.build_messages(cc.segment_slice(rows, segs[1]), 5)
    assert msgs[0]["t"] == "user" and msgs[0]["text"].startswith("This session is being continued")


def test_system_prompt_flags_reconstruction():
    rows = _rows(("init", "system", '{"model":"claude-opus-4-5-20251101","tools":["Bash"],"cwd":"/w"}'))
    sp = cc.system_prompt(rows)
    assert "RECONSTRUCTION NOTE" in sp and "AI Village" in sp

"""tierb_lib against the shapes the AI Village export really stores: model outputs as each provider
returns them (the structure of real rows, with synthetic text), timestamps as Postgres writes them, and one
small synthetic events file run through the loader so the chat AND activity paths are both covered.

test_real_events_file runs only with VILLAGE_DATA_TESTS=1 and VILLAGE_DATASET set (one streaming pass
over events.jsonl.gz)."""

import gzip
import json
import os

import pytest

from village_introspect import tierb_lib as tb
from village_introspect.cc_lib import DATASET

SIG = "[BLOB_REMOVED]"
# (label, output, expected (text, kind)): synthetic text, shaped like real rows of each kind
SHAPES = [
    (
        "anthropic bare block list (Claude Opus 4.5 / Haiku 4.5 / Sonnet 4.5 / 3.7 Sonnet AGENT_TALK)",
        [
            {"type": "thinking", "thinking": "Checking where things stand:\n- It is mid-morning", "signature": SIG},
            {"type": "text", "text": "Sending the room a brief note."},
            {"id": "toolu_1", "name": "send_message_to_chat", "type": "tool_use", "input": {"message": "hi"}},
        ],
        ("Checking where things stand:\n- It is mid-morning", "verbatim"),
    ),
    (
        "anthropic bare block list without text (a WAIT activity)",
        [
            {"type": "thinking", "thinking": "Nothing to do yet, so wait.", "signature": SIG},
            {"id": "toolu_2", "name": "wait", "type": "tool_use", "input": {}},
        ],
        ("Nothing to do yet, so wait.", "verbatim"),
    ),
    (
        "anthropic full message (a PAUSE activity)",
        {
            "id": "msg_1",
            "role": "assistant",
            "type": "message",
            "model": "claude-haiku-4-5-20251001",
            "content": [
                {"type": "thinking", "thinking": "The next check-in is later today.", "signature": SIG},
                {"id": "toolu_3", "name": "pause", "type": "tool_use", "input": {"seconds": 900}},
            ],
            "stop_reason": "tool_use",
        },
        ("The next check-in is later today.", "verbatim"),
    ),
    (
        "anthropic full message, thinking display omitted (empty)",
        {
            "role": "assistant",
            "type": "message",
            "model": "claude-opus-4-7",
            "content": [
                {"type": "thinking", "thinking": "", "signature": SIG},
                {"id": "toolu_4", "name": "send_message_to_chat", "type": "tool_use", "input": {"message": "x"}},
            ],
        },
        ("", ""),
    ),
    (
        "chat completions reasoning_content (DeepSeek-V3.2 / Kimi, content empty, tool call)",
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "send_message_to_chat"}}],
            "reasoning_content": "I have been tracking the shared task list.",
        },
        ("I have been tracking the shared task list.", "reasoning"),
    ),
    (
        "chat completions reasoning_content with refusal field (Grok 4.5)",
        {
            "role": "assistant",
            "content": "The page is up.",
            "refusal": None,
            "tool_calls": [],
            "reasoning_content": "The page is up, so announce it briefly.",
        },
        ("The page is up, so announce it briefly.", "reasoning"),
    ),
    (
        "OpenRouter reasoning + reasoning_details (GLM-5.2 / DeepSeek-V4-Pro)",
        {
            "role": "assistant",
            "content": "Two edits are still pending.",
            "refusal": None,
            "reasoning": "Where things stand:\n1. The draft needs two edits",
            "reasoning_details": [{"text": "Where things stand:\n1. The draft needs two edits"}],
            "tool_calls": [],
        },
        ("Where things stand:\n1. The draft needs two edits", "reasoning"),
    ),
    (
        "OpenRouter encrypted reasoning only (Muse Spark): nothing readable",
        {
            "role": "assistant",
            "content": None,
            "refusal": None,
            "reasoning": None,
            "reasoning_details": [{"id": "rs_1", "data": "ENCRYPTED"}],
            "tool_calls": [],
        },
        ("", ""),
    ),
    (
        "chat completions empty reasoning_content",
        {"role": "assistant", "content": "Pausing for a while.", "tool_calls": [], "reasoning_content": ""},
        ("", ""),
    ),
    (
        "OpenAI chat completion, no reasoning (o3 / GPT-4o)",
        {"role": "assistant", "content": "A quick recap.", "refusal": None, "annotations": []},
        ("", ""),
    ),
    (
        "OpenAI Responses items with a reasoning summary",
        [
            {
                "id": "rs_1",
                "type": "reasoning",
                "summary": [{"text": "**Fixture reasoning summary**", "type": "summary_text"}],
            },
            {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
        ],
        ("**Fixture reasoning summary**", "summary"),
    ),
    (
        "OpenAI Responses items, encrypted reasoning only",
        [
            {"id": "rs_2", "type": "reasoning", "content": [], "summary": [], "encrypted_content": "ENCRYPTED"},
            {"type": "function_call", "name": "send_message_to_chat", "arguments": "{}"},
        ],
        ("", ""),
    ),
    (
        "Gemini thought part",
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "**Plan for this step**", "thought": True}, {"text": "ok"}],
                    },
                }
            ],
            "modelVersion": "gemini-2.5-pro",
        },
        ("**Plan for this step**", "verbatim"),
    ),
]


@pytest.mark.parametrize("label, output, expected", SHAPES, ids=[s[0] for s in SHAPES])
def test_extract_thinking_real_shapes(label, output, expected):
    assert tb.extract_thinking(output) == expected


def _gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_loader_extracts_thinking_for_chat_and_activity(tmp_path, monkeypatch):
    """The one events pass gives chat and activity records the same thinking extraction."""
    _gz(
        tmp_path / "agents.jsonl.gz",
        [
            {"id": "op45", "name": "Claude Opus 4.5", "model_string": "claude-opus-4-5-20251101"},
            {"id": "ds", "name": "DeepSeek-V3.2", "model_string": "deepseek-reasoner"},
            {"id": "glm", "name": "GLM-5.2", "model_string": "z-ai/glm-5.2"},
        ],
    )
    bare, wait, dsk, glm = SHAPES[0][1], SHAPES[1][1], SHAPES[4][1], SHAPES[6][1]

    def row(ei, action, **data):  # a 5-digit fraction, as Postgres trims trailing zeros
        return {"event_index": ei, "created_at": "2026-06-22 17:05:00.12345", "data": {"actionType": action, **data}}

    events = [
        row(1, "USER_TALK", roomId="R", speakerName="automated", content="time to start the day"),
        row(2, "AGENT_TALK", roomId="R", speakerId="op45", content="hi", output=bare),
        row(3, "AGENT_TALK", roomId="R", speakerId="ds", content="report", output=dsk),
        row(4, "CONSOLIDATE", roomId="R", agentId="ds", output=dsk),
        row(5, "WAIT", roomId="R", agentId="op45", output=wait),
        row(6, "START_USING_COMPUTER", roomId="R", agentId="glm", output=glm),
    ]
    _gz(tmp_path / "events.jsonl.gz", events)
    monkeypatch.setattr(tb, "_ROSTER", None)
    monkeypatch.setattr(tb, "_ACTIVITY", None)
    monkeypatch.setattr(tb, "_ROOM_NAMES", {})
    monkeypatch.setattr(tb, "village_day", lambda ca: 5)
    chat = tb._load_chat_uncached(tmp_path)
    kinds = {r["ei"]: (r["think_kind"], r["thinking"][:12]) for r in chat}
    assert kinds == {1: ("", ""), 2: ("verbatim", "Checking whe"), 3: ("reasoning", "I have been ")}
    acts = {a["ei"]: (a["type"], a["think_kind"], a["thinking"][:12]) for a in tb._ACTIVITY}
    assert acts == {
        4: ("CONSOLIDATE", "reasoning", "I have been "),
        5: ("WAIT", "verbatim", "Nothing to d"),
        6: ("START_USING_COMPUTER", "reasoning", "Where things"),
    }
    assert all(tb.parse_ts(r["created_at"]) is not None for r in chat)
    # the idle-nudger is a bot in the prompt, not a human
    monkeypatch.setattr(tb, "_ROSTER", {"op45": {"agent_id": "op45", "name": "Claude Opus 4.5"}})
    pp = tb.build_probe_prompt("op45", 2, "", window=99, chat=chat, fmt="transcript")
    assert pp["messages"][0]["text"].startswith("automated [automated]: time to start the day")


@pytest.mark.skipif(
    os.environ.get("VILLAGE_DATA_TESTS") != "1" or not (DATASET / "events.jsonl.gz").exists(),
    reason="set VILLAGE_DATA_TESTS=1 and VILLAGE_DATASET (streams the real events file)",
)
def test_real_events_file():
    """Every created_at parses, and every output that carries thinking text yields it."""
    unparsed, missed, total = [], [], 0
    with gzip.open(DATASET / "events.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if tb.parse_ts(r.get("created_at", "")) is None:
                unparsed.append(r.get("created_at"))
            o = (r.get("data") or {}).get("output")
            blocks = o if isinstance(o, list) else (o.get("content") if isinstance(o, dict) else None)
            has = any(
                isinstance(b, dict) and b.get("type") == "thinking" and (b.get("thinking") or "").strip()
                for b in (blocks if isinstance(blocks, list) else [])
            ) or (
                isinstance(o, dict)
                and any(isinstance(o.get(k), str) and o[k].strip() for k in ("reasoning", "reasoning_content"))
            )
            total += has
            if has and not tb.extract_thinking(o)[0]:
                missed.append(r.get("event_index"))
    print(f"\n{total} outputs carry thinking text; {len(missed)} not extracted; {len(unparsed)} unparsed timestamps")
    assert not unparsed, unparsed[:5]
    assert not missed, missed[:5]

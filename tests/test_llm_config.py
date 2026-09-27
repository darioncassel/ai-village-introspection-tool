"""Unit tests for provider routing, API-key handling, the provider clients (with fake SDK clients, so no
network or keys are needed), the model picker, and the date -> day map cache.

The fake clients bind every call to the installed SDK's real method signature, so a keyword the SDK doesn't
take (such as temperature on anthropic 1.x) fails here as it would in a live call. The wire tests go one step
further and run the real SDK against a mock HTTP transport."""

import inspect
import json
import types

import anthropic
import openai
import pytest
from anthropic.resources.messages import Messages
from openai.resources.chat.completions import Completions

from village_introspect import config, llm, server

KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
MSGS = [{"t": "user", "text": "hi"}]
_ANTHROPIC_STREAM = inspect.signature(Messages.stream)
_OPENAI_CREATE = inspect.signature(Completions.create)


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("VILLAGE_MODELS", raising=False)


def _http(sdk):
    """The HTTP library the installed SDK is built on (httpx2 for anthropic 1.x and openai 3.x)."""
    base = sdk._base_client
    return getattr(base, "httpx2", None) or base.httpx


def _bad_request(sdk, message):
    """A real SDK 400, as the API returns it for a setting the model doesn't support."""
    http = _http(sdk)
    resp = http.Response(400, request=http.Request("POST", "https://api.example.invalid/v1"))
    return sdk.BadRequestError(message, response=resp, body=None)


def test_provider_for():
    assert llm.provider_for("claude-opus-4-8") == "anthropic"
    assert llm.provider_for("claude-3-5-sonnet-20241022") == "anthropic"
    assert llm.provider_for("gpt-5.5") == "openai"
    assert llm.provider_for("o3") == "openai"
    assert llm.provider_for("chatgpt-4o-latest") == "openai"
    assert llm.provider_for("codex-mini-latest") == "openai"
    assert llm.provider_for("o4-mini") == "openai"
    assert llm.provider_for("ft:gpt-4o-mini-2024-07-18:org::abc123") == "openai"
    # another vendor's bare id goes nowhere: OpenAI doesn't serve it, and OpenRouter needs its vendor/model id
    for bare in ("grok-4", "gemini-3.1-pro", "deepseek-v4-pro", "kimi-k3", "glm-5.2", "o", "orca"):
        assert llm.provider_for(bare) is None, bare
    assert llm.provider_for("google/gemini-3.1-pro-preview") == "openrouter"
    assert llm.provider_for("openai/gpt-5.5") == "openrouter"
    assert llm.provider_for("anthropic/claude-sonnet-4.5") == "openrouter"


def test_missing_key_names_the_variable():
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        llm.make_client("x-ai/grok-4.5")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm.make_client("claude-opus-4-8")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm.make_client("chatgpt-4o-latest")


def test_unroutable_model_is_refused_with_a_hint():
    with pytest.raises(ValueError, match=r"OpenRouter id \(vendor/model, e\.g\. x-ai/grok-4\)"):
        llm.make_client("grok-4")


# ------------------------------------------------------------------------------------------------- OpenAI
class _FakeOpenAI:
    last = None
    rejects = ()  # kwargs to answer with a 400 naming them
    reply = {}  # message/choice overrides

    def __init__(self, **kw):
        self.init = kw
        self.calls = []
        _FakeOpenAI.last = self
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        _OPENAI_CREATE.bind(self, **kw)
        self.calls.append(json.loads(json.dumps(kw)))
        self.kw = kw
        for k in _FakeOpenAI.rejects:
            if k in kw:
                raise _bad_request(openai, f"Unsupported parameter: '{k}' is not supported with this model.")
        r = dict(content="the answer", reasoning="some reasoning", refusal=None, finish_reason="stop")
        r.update(_FakeOpenAI.reply)
        finish = r.pop("finish_reason")
        msg = types.SimpleNamespace(**r)
        usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=7)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=finish)], usage=usage)


@pytest.fixture()
def fake_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(_FakeOpenAI, "rejects", ())
    monkeypatch.setattr(_FakeOpenAI, "reply", {})
    return _FakeOpenAI


def test_fake_openai_rejects_unknown_kwargs(fake_openai):
    with pytest.raises(TypeError):
        fake_openai()._create(model="m", messages=[], not_a_real_setting=1)


def test_openrouter_client_uses_base_url_env_key_and_reasoning(fake_openai):
    out = llm.call(
        "SYS", [{"t": "user", "text": "hi"}, {"t": "assistant", "text": ""}], "google/gemini-3.1-pro-preview"
    )
    fake = fake_openai.last
    assert fake.init == {"base_url": llm.OPENROUTER_BASE_URL, "api_key": "or-test"}
    assert fake.kw["messages"][0] == {"role": "system", "content": "SYS"}
    assert fake.kw["messages"][2] == {"role": "assistant", "content": "(no textual response)"}
    assert fake.kw["extra_body"] == {"reasoning": {"effort": "high"}}
    assert "temperature" not in fake.kw and "reasoning_effort" not in fake.kw
    assert out == {
        "thinking": "some reasoning",
        "text": "the answer",
        "tool_use": None,
        "usage": {"in": 11, "out": 7},
        "stop_reason": "end",
        "request": {"provider": "openrouter", "thinking": None, "effort": "high", "max_tokens": 16000, "fallbacks": []},
    }


def test_openai_client_sends_reasoning_effort_not_temperature(fake_openai):
    out = llm.call("SYS", MSGS, "gpt-5.5")
    fake = fake_openai.last
    assert fake.init == {"api_key": "oa-test"}
    assert fake.kw["reasoning_effort"] == "high" and fake.kw["max_completion_tokens"] == 128000
    assert "temperature" not in fake.kw and "extra_body" not in fake.kw
    assert out["request"] == {
        "provider": "openai",
        "thinking": None,
        "effort": "high",
        "max_tokens": 128000,
        "fallbacks": [],
    }


def test_openai_drops_reasoning_effort_if_rejected(fake_openai):
    fake_openai.rejects = ("reasoning_effort",)  # a non-reasoning model such as chatgpt-4o-latest
    out = llm.call("SYS", MSGS, "chatgpt-4o-latest")
    first, retry = fake_openai.last.calls
    assert first["reasoning_effort"] == "high" and "reasoning_effort" not in retry and "temperature" not in retry
    assert out["request"]["effort"] is None and out["request"]["fallbacks"] == ["dropped effort"]
    assert out["text"] == "the answer"


@pytest.mark.parametrize(
    "reply, text, stop",
    [
        ({"content": None, "refusal": "I can't help with that."}, "I can't help with that.", "refusal"),
        ({"content": "Honestly, when I read that I felt", "finish_reason": "length"}, None, "max_tokens"),
        ({"content": None, "finish_reason": "length"}, "", "max_tokens"),
        ({"content": "", "finish_reason": "content_filter"}, "", "refusal"),
        ({"finish_reason": "error"}, None, "error"),
    ],
)
def test_openai_stop_reason_and_refusal(fake_openai, reply, text, stop):
    fake_openai.reply = reply
    out = llm.call("SYS", MSGS, "moonshotai/kimi-k3")
    assert out["stop_reason"] == stop
    assert out["text"] == (reply.get("content", "the answer") if text is None else text)


# ---------------------------------------------------------------------------------------------- Anthropic
_REJECT_MSG = {
    "effort": "output_config.effort: This model does not support the effort parameter.",
    "adaptive": "thinking.type: adaptive thinking is not supported on this model",
    "thinking": "thinking: extended thinking is not supported on this model",
}


class _FakeAnthropic:
    last = None
    rejects = ()  # "effort" / "adaptive" / "thinking", checked in this order: answer with a 400
    stop_reason = "end_turn"

    def __init__(self, **kw):
        self.init = kw
        self.calls = []
        _FakeAnthropic.last = self
        self.messages = types.SimpleNamespace(stream=self._stream)

    def _stream(self, **kw):
        _ANTHROPIC_STREAM.bind(self, **kw)
        self.calls.append(json.loads(json.dumps(kw)))
        thinking = kw.get("thinking") or {}
        present = {
            "effort": "output_config" in kw,
            "adaptive": thinking.get("type") == "adaptive",
            "thinking": thinking,
        }
        for r in _FakeAnthropic.rejects:
            if present[r]:
                raise _bad_request(anthropic, _REJECT_MSG[r])
        blocks = [
            types.SimpleNamespace(type="thinking", thinking="summary of thinking"),
            types.SimpleNamespace(type="text", text="in character"),
        ]
        final = types.SimpleNamespace(
            content=blocks,
            usage=types.SimpleNamespace(input_tokens=5, output_tokens=3),
            stop_reason=_FakeAnthropic.stop_reason,
        )

        class _Ctx:
            def __enter__(self_inner):
                return types.SimpleNamespace(get_final_message=lambda: final)

            def __exit__(self_inner, *a):
                return False

        return _Ctx()


@pytest.fixture()
def fake_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    monkeypatch.setattr(_FakeAnthropic, "rejects", ())
    monkeypatch.setattr(_FakeAnthropic, "stop_reason", "end_turn")
    return _FakeAnthropic


def test_fake_anthropic_rejects_temperature(fake_anthropic):
    # the kwarg the old fallback added: the 1.x SDK has no temperature argument, and the fake must say so
    with pytest.raises(TypeError):
        fake_anthropic()._stream(model="m", max_tokens=10, messages=[], temperature=1.0)


@pytest.mark.parametrize(
    "model, version",
    [
        ("claude-opus-4-8", (4, 8)),
        ("claude-opus-4-5-20251101", (4, 5)),
        ("claude-haiku-4-5-20251001", (4, 5)),
        ("claude-sonnet-4-20250514", (4, 0)),
        ("claude-opus-5", (5, 0)),
        ("claude-fable-5-1", (5, 1)),
        ("claude-3-5-sonnet-20241022", (3, 5)),
        ("claude-next", None),
    ],
)
def test_claude_version(model, version):
    assert llm._claude_version(model) == version


@pytest.mark.parametrize(
    "model", ["claude-opus-4-8", "claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-5", "claude-next"]
)
def test_anthropic_adaptive_generation(fake_anthropic, model):
    out = llm.call("SYS", MSGS, model)
    fake = fake_anthropic.last
    assert fake.init == {"api_key": "sk-test"}
    (kw,) = fake.calls
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kw["output_config"] == {"effort": "high"}
    assert "temperature" not in kw and "extra_body" not in kw
    assert kw["system"] == "SYS" and kw["max_tokens"] == llm.max_output_for(model)
    assert out["thinking"] == "summary of thinking" and out["text"] == "in character"
    assert out["stop_reason"] == "end"
    assert out["request"] == {
        "provider": "anthropic",
        "thinking": {"type": "adaptive", "display": "summarized"},
        "effort": "high",
        "max_tokens": llm.max_output_for(model),
        "fallbacks": [],
    }


@pytest.mark.parametrize("model", ["claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"])
def test_anthropic_budget_generation(fake_anthropic, model):
    fake_anthropic.rejects = ("adaptive", "effort")  # as these models do: neither is ever sent
    out = llm.call("SYS", MSGS, model)
    (kw,) = fake_anthropic.last.calls
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 8000}
    assert kw["thinking"]["budget_tokens"] < kw["max_tokens"] == 16000
    assert "output_config" not in kw and "temperature" not in kw and "extra_body" not in kw
    assert out["request"]["thinking"] == kw["thinking"] and out["request"]["effort"] is None
    assert out["request"]["fallbacks"] == [] and out["thinking"] == "summary of thinking"


@pytest.mark.parametrize("rejects", [("effort", "adaptive"), ("adaptive", "effort")])
def test_anthropic_fallback_is_cumulative(fake_anthropic, rejects):
    # a model the table gets wrong rejects both settings, reporting either one first
    fake_anthropic.rejects = rejects
    out = llm.call("SYS", MSGS, "claude-next")
    calls = fake_anthropic.last.calls
    assert len(calls) == 3
    assert "output_config" not in calls[-1] and "thinking" not in calls[-1] and "temperature" not in calls[-1]
    assert sorted(out["request"]["fallbacks"]) == ["dropped effort", "dropped thinking"]
    assert out["request"]["thinking"] is None and out["request"]["effort"] is None
    assert out["text"] == "in character"


def test_anthropic_budget_thinking_dropped_if_rejected(fake_anthropic):
    fake_anthropic.rejects = ("thinking",)  # e.g. a Claude 3 model with no extended thinking
    out = llm.call("SYS", MSGS, "claude-3-5-haiku-20241022")
    first, retry = fake_anthropic.last.calls
    assert first["thinking"]["type"] == "enabled" and "thinking" not in retry and "temperature" not in retry
    assert out["request"]["fallbacks"] == ["dropped thinking"]


def test_anthropic_fallback_is_bounded(fake_anthropic, monkeypatch):
    def always_400(self, kw):
        fake_anthropic.last.calls.append(dict(kw))
        raise _bad_request(anthropic, "thinking and effort are both invalid here")

    monkeypatch.setattr(llm.AnthropicClient, "_final", always_400)
    with pytest.raises(anthropic.BadRequestError):
        llm.call("SYS", MSGS, "claude-opus-4-8")
    assert len(fake_anthropic.last.calls) == 3  # the original, then one per setting dropped


def test_anthropic_other_errors_are_not_fallbacks(fake_anthropic, monkeypatch):
    def boom(self, kw):
        fake_anthropic.last.calls.append(dict(kw))
        raise RuntimeError("unexpected effort failure")  # no 400: not a rejected setting

    monkeypatch.setattr(llm.AnthropicClient, "_final", boom)
    with pytest.raises(RuntimeError):
        llm.call("SYS", MSGS, "claude-opus-4-8")
    assert len(fake_anthropic.last.calls) == 1


@pytest.mark.parametrize(
    "raw, norm",
    [("end_turn", "end"), ("max_tokens", "max_tokens"), ("refusal", "refusal"), ("pause_turn", "pause_turn")],
)
def test_anthropic_stop_reason(fake_anthropic, raw, norm):
    fake_anthropic.stop_reason = raw
    assert llm.call("SYS", MSGS, "claude-opus-4-8")["stop_reason"] == norm


def test_default_models_send_no_temperature(fake_anthropic, fake_openai):
    for model, _ in config.DEFAULT_MODELS:
        out = llm.call("SYS", MSGS, model)
        fake = fake_anthropic.last if llm.provider_for(model) == "anthropic" else fake_openai.last
        assert all("temperature" not in kw for kw in fake.calls), model
        assert out["request"]["provider"] == llm.provider_for(model)


# ------------------------------------------------------------------------------ wire (real SDK, mock HTTP)
def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def test_anthropic_wire_request_and_stream(monkeypatch):
    bodies = []
    http = _http(anthropic)

    def handler(request):
        bodies.append(json.loads(request.content))
        msg = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-4-8", "content": []}
        msg.update(stop_reason=None, stop_sequence=None, usage={"input_tokens": 5, "output_tokens": 1})
        events = [
            {"type": "message_start", "message": msg},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Honestly, I"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 9}},
            {"type": "message_stop"},
        ]
        return http.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(events))

    real = anthropic.Anthropic
    client = http.Client(transport=http.MockTransport(handler))
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: real(http_client=client, max_retries=0, **kw))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    out = llm.call("SYS", MSGS, "claude-opus-4-8")
    (body,) = bodies
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"} and "temperature" not in body
    assert out["text"] == "Honestly, I" and out["thinking"] == "hmm" and out["stop_reason"] == "max_tokens"


def test_openai_wire_request(monkeypatch):
    bodies = []
    http = _http(openai)

    def handler(request):
        bodies.append(json.loads(request.content))
        msg = {"role": "assistant", "content": None, "refusal": "I can't help with that."}
        body = {"id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-5.5"}
        body["choices"] = [{"index": 0, "message": msg, "finish_reason": "stop"}]
        return http.Response(200, json=body)

    real = openai.OpenAI
    client = http.Client(transport=http.MockTransport(handler))
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: real(http_client=client, max_retries=0, **kw))
    monkeypatch.setenv("OPENAI_API_KEY", "oa-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    out = llm.call("SYS", MSGS, "gpt-5.5")
    llm.call("SYS", MSGS, "x-ai/grok-4.5")
    oa, orouter = bodies
    assert oa["reasoning_effort"] == "high" and "temperature" not in oa and "reasoning" not in oa
    assert orouter["reasoning"] == {"effort": "high"} and "temperature" not in orouter
    assert out["text"] == "I can't help with that." and out["stop_reason"] == "refusal"


def test_model_picker_marks_usable_by_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    ms = {m["id"]: m for m in server._models()}
    assert ms["google/gemini-3.1-pro-preview"]["usable"]
    assert not ms["claude-opus-4-8"]["usable"]
    assert ms["claude-opus-4-8"]["note"].startswith("set ANTHROPIC_API_KEY")
    assert "via OpenRouter" in ms["x-ai/grok-4.5"]["note"]


def test_model_picker_greys_out_unroutable_ids(monkeypatch):
    monkeypatch.setenv("VILLAGE_MODELS", "grok-4,x-ai/grok-4.5,gpt-5.5")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-test")
    ms = {m["id"]: m for m in server._models()}
    assert not ms["grok-4"]["usable"] and ms["grok-4"]["provider"] is None
    assert ms["grok-4"]["note"] == "unknown provider: use its OpenRouter id (vendor/model)"
    assert ms["x-ai/grok-4.5"]["usable"] and ms["gpt-5.5"]["usable"]


def test_village_models_env_replaces_list(monkeypatch):
    monkeypatch.setenv("VILLAGE_MODELS", "claude-sonnet-4-6, deepseek/deepseek-v4-pro ,")
    assert config.configured_models() == [("claude-sonnet-4-6", ""), ("deepseek/deepseek-v4-pro", "")]


def test_day_ranges_cached_and_invalidated(tmp_path, monkeypatch):
    """The day ranges are built once, read back from the cache (a rebuild is made to fail), and rebuilt
    when the transcript's size or mtime changes."""
    import os

    ds, state = tmp_path / "ds", tmp_path / "state"
    ds.mkdir()
    tr = ds / "village-transcript.json"

    def write(day, ts):
        tr.write_text(json.dumps({"days": [{"date": ts[:10], "day": day, "events": [{"timestamp": ts}]}]}))

    write(1, "2025-04-02T17:00:00.000Z")
    monkeypatch.setattr(config, "STATE_DIR", state)
    first = config.day_ranges(ds)
    assert [(r["day"], r["date"]) for r in first] == [(1, "2025-04-02")]
    assert (state / "cache" / "day_ranges.json").is_file()
    build = config._ranges_from_transcript

    def no_rebuild(t):
        raise AssertionError("rebuilt instead of reading the cache")

    monkeypatch.setattr(config, "_ranges_from_transcript", no_rebuild)
    assert config.day_ranges(ds) == first  # served from the cache
    st = tr.stat()
    write(2, "2025-04-03T17:00:00.000Z")  # same size, different content, mtime restored: still cached
    os.utime(tr, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert config.day_ranges(ds) == first
    monkeypatch.setattr(config, "_ranges_from_transcript", build)
    os.utime(tr, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # mtime changes -> rebuilt
    assert [r["day"] for r in config.day_ranges(ds)] == [2]
    write(12, "2025-04-14T17:00:00.000Z")  # size changes -> rebuilt
    assert [r["day"] for r in config.day_ranges(ds)] == [12]


def test_default_model_is_first_village_models_entry(monkeypatch):
    assert config.preferred_default() == "claude-opus-4-8"
    monkeypatch.setenv("VILLAGE_MODELS", "claude-sonnet-4-6,claude-opus-4-8")
    assert config.preferred_default() == "claude-sonnet-4-6"

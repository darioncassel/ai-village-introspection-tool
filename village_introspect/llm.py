"""Live model calls for probes: Anthropic, OpenAI, and OpenRouter (through its OpenAI-compatible API).

Keys come only from the environment: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY. The provider
is inferred from the model id (provider_for). Messages use a neutral format, translated per provider:

    {"t": "user", "text": str}
    {"t": "assistant", "text": str}
"""

from __future__ import annotations

import dataclasses
import os
import random
import re
import time
from typing import Optional

PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_RETRY_ATTEMPTS = 8  # capped exponential backoff: rides out ~3 min of overload / rate limiting


_OPENAI_ID = re.compile(r"^(gpt-|chatgpt-|codex-|ft:|o\d)")


def provider_for(model: str) -> Optional[str]:
    """Which API serves a model id, or None if it can't be routed. "vendor/model" ids go to OpenRouter, ids
    starting with "claude" to Anthropic, and OpenAI's own ids (gpt-*, o1/o3/o4-*, chatgpt-*, codex-*, ft:*)
    to OpenAI. Any other bare id, such as grok-4 or gemini-3.1-pro, is another vendor's model: it is reached
    through OpenRouter under its vendor/model id (e.g. x-ai/grok-4), which can't be guessed reliably."""
    if "/" in model:
        return "openrouter"
    if model.startswith("claude"):
        return "anthropic"
    if _OPENAI_ID.match(model):
        return "openai"
    return None


def unroutable_reason(model: str) -> str:
    return (
        f"can't tell which provider serves {model!r}: use its OpenRouter id (vendor/model, e.g. x-ai/grok-4), "
        "a claude-* id for Anthropic, or an OpenAI id (gpt-*, o*, chatgpt-*, codex-*)"
    )


def has_key(provider: str) -> bool:
    return bool(os.environ.get(PROVIDER_KEYS[provider]))


def _require_key(provider: str, model: str) -> str:
    var = PROVIDER_KEYS[provider]
    key = os.environ.get(var)
    if not key:
        raise RuntimeError(f"{var} is not set; export it to ask {model}")
    return key


def _is_transient(e: Exception) -> bool:
    code = getattr(e, "status_code", None)
    s = str(e).lower()
    return code in (408, 409, 425, 429, 500, 502, 503, 529) or any(
        k in s for k in ("overloaded", "rate limit", "timeout", "timed out", "temporarily", "connection")
    )


def _retry(fn):
    """Retry a single call on transient API errors (overload / rate limit / timeout) with jittered backoff."""
    for i in range(_RETRY_ATTEMPTS):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not _is_transient(e) or i == _RETRY_ATTEMPTS - 1:
                raise
            time.sleep(min(2**i, 30) + random.uniform(0, 1.5))


def _with_fallbacks(send, kw: dict, steps: tuple, fallbacks: list):
    """send(kw) with retries; if the model rejects a setting with a 400 that names it, drop that setting
    and try again. steps = ((label, kwarg, words in the error), ...); each step runs at most once and
    they accumulate, so this ends after len(steps) extra calls. The labels dropped go into fallbacks."""
    while True:
        try:
            return _retry(lambda: send(kw))
        except Exception as e:  # noqa: BLE001
            if getattr(e, "status_code", None) != 400:
                raise
            msg = str(e).lower()
            step = next((st for st in steps if st[1] in kw and any(w in msg for w in st[2])), None)
            if step is None:
                raise
            kw.pop(step[1])
            fallbacks.append(f"dropped {step[0]}")


@dataclasses.dataclass
class Completion:
    text: str
    thinking: str
    usage: dict
    stop_reason: Optional[str] = None  # "end", "max_tokens", "refusal", or the provider's raw value
    request: dict = dataclasses.field(default_factory=dict)  # the settings actually sent


EFFORT = "high"  # asked of every provider, so answers from different models are comparable

# Per-model max output tokens: a ceiling, not a target. A low cap can be used up entirely by reasoning,
# which leaves an empty answer. Unlisted models get the conservative default.
_MODEL_MAX_OUTPUT = {"claude-opus-4-8": 128000, "gpt-5.5": 128000}
_DEFAULT_MAX_OUTPUT = 16000


def max_output_for(model: str) -> int:
    return _MODEL_MAX_OUTPUT.get(model, _DEFAULT_MAX_OUTPUT)


def _alternating(messages: list) -> list:
    return [
        {
            "role": "assistant" if m["t"] == "assistant" else "user",
            "content": m["text"] or ("(no textual response)" if m["t"] == "assistant" else "(empty)"),
        }
        for m in messages
    ]


def _stop(raw: Optional[str], mapping: dict) -> Optional[str]:
    return mapping.get(raw, raw) if raw else None


# Adaptive thinking and the effort setting start with the 4.6 generation (Opus 4.6, Sonnet 4.6) and cover
# every later one. Older models (Opus 4.5, Haiku 4.5, Sonnet 4.5, ...) take a fixed thinking budget
# instead. Haiku 4.5 and Sonnet 4.5 reject effort and "high" is already Opus 4.5's default, so effort is not
# sent to that generation.
_ADAPTIVE_FROM = (4, 6)
_THINKING_BUDGET = 8000  # budget_tokens for the older generation (the API needs 1024 <= budget < max_tokens)
_ANTHROPIC_STOP = {"end_turn": "end", "max_tokens": "max_tokens", "refusal": "refusal"}


def _claude_version(model: str) -> Optional[tuple]:
    """(major, minor) from a Claude model id: claude-opus-4-5-20251101 -> (4, 5), claude-opus-5 -> (5, 0),
    claude-3-5-sonnet-20241022 -> (3, 5). None if the id doesn't follow either naming scheme."""
    m = re.match(r"claude-(?:[a-z]+-)?(\d+)(?:-(\d{1,2}))?(?!\d)", model)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else None


def anthropic_thinking(model: str, max_tokens: int, effort: Optional[str] = EFFORT) -> tuple:
    """(thinking, effort) to send to a Claude model. Unrecognised ids are treated as the newest generation;
    if a model rejects either setting, AnthropicClient drops it and retries."""
    version = _claude_version(model)
    if version is None or version >= _ADAPTIVE_FROM:
        # display "summarized": Opus 4.8 defaults to "omitted", i.e. EMPTY thinking text; a summary is
        # the most any current model returns (the raw chain of thought is never returned)
        return {"type": "adaptive", "display": "summarized"}, effort
    return {"type": "enabled", "budget_tokens": min(_THINKING_BUDGET, max_tokens // 2)}, None


class AnthropicClient:
    # what a 400 may make _with_fallbacks drop: the setting the error names (effort first if it names both),
    # each at most once, cumulatively
    FALLBACKS = (("effort", "output_config", ("effort", "output_config")), ("thinking", "thinking", ("thinking",)))

    def __init__(self, model: str) -> None:
        import anthropic

        self.model = model
        self._c = anthropic.Anthropic(api_key=_require_key("anthropic", model))

    def _final(self, kw: dict):
        # stream rather than create(): the SDK refuses a non-streaming request whose max_tokens could run
        # past its 10-minute limit, and the Opus cap is 128k. get_final_message() returns the same Message.
        with self._c.messages.stream(**kw) as st:
            return st.get_final_message()

    def complete(self, system: str, messages: list, max_tokens: int, effort: Optional[str] = EFFORT) -> Completion:
        # no temperature: the 1.x SDK has no such argument, 1.0 is the API default, and 4.7+ models reject it
        kw = dict(model=self.model, max_tokens=max_tokens, system=system, messages=_alternating(messages))
        thinking, effort = anthropic_thinking(self.model, max_tokens, effort)
        kw["thinking"] = thinking
        if effort:
            kw["output_config"] = {"effort": effort}
        fallbacks: list = []
        r = _with_fallbacks(self._final, kw, self.FALLBACKS, fallbacks)
        text, thinking_text = "", ""
        for b in r.content:
            if b.type == "text":
                text += b.text
            elif b.type == "thinking":
                thinking_text += getattr(b, "thinking", "") or ""
        request = {
            "provider": "anthropic",
            "thinking": kw.get("thinking"),
            "effort": (kw.get("output_config") or {}).get("effort"),
            "max_tokens": max_tokens,
            "fallbacks": fallbacks,
        }
        usage = {"in": r.usage.input_tokens, "out": r.usage.output_tokens}
        return Completion(text, thinking_text, usage, _stop(getattr(r, "stop_reason", None), _ANTHROPIC_STOP), request)


_OPENAI_STOP = {"stop": "end", "length": "max_tokens", "content_filter": "refusal"}


class OpenAIClient:
    """OpenAI, or OpenRouter through its OpenAI-compatible endpoint."""

    def __init__(self, model: str, provider: str = "openai") -> None:
        import openai

        self.model = model
        self.provider = provider
        key = _require_key(provider, model)
        if provider == "openrouter":
            self._c = openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
        else:
            self._c = openai.OpenAI(api_key=key)

    def complete(self, system: str, messages: list, max_tokens: int, effort: Optional[str] = EFFORT) -> Completion:
        # no temperature: 1.0 is the default, and OpenAI reasoning models reject it
        kw = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}] + _alternating(messages),
            max_completion_tokens=max_tokens,
        )
        steps: tuple = ()
        if effort and self.provider == "openrouter":
            # OpenRouter's unified reasoning setting, mapped to each upstream model's own control
            kw["extra_body"] = {"reasoning": {"effort": effort}}
            steps = (("effort", "extra_body", ("reasoning",)),)
        elif effort:
            kw["reasoning_effort"] = effort  # rejected by non-reasoning models (gpt-4o, ...): dropped then
            steps = (("effort", "reasoning_effort", ("reasoning_effort",)),)
        fallbacks: list = []
        r = _with_fallbacks(lambda k: self._c.chat.completions.create(**k), kw, steps, fallbacks)
        choice = r.choices[0]
        msg = choice.message
        refusal = getattr(msg, "refusal", None) or ""
        stop = "refusal" if refusal else _stop(getattr(choice, "finish_reason", None), _OPENAI_STOP)
        usage = {}
        if getattr(r, "usage", None):
            usage = {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}
        request = {
            "provider": self.provider,
            "thinking": None,
            "effort": None if fallbacks else effort,
            "max_tokens": max_tokens,
            "fallbacks": fallbacks,
        }
        return Completion(msg.content or refusal, getattr(msg, "reasoning", None) or "", usage, stop, request)


def make_client(model: str):
    provider = provider_for(model)
    if provider is None:
        raise ValueError(unroutable_reason(model))
    return AnthropicClient(model) if provider == "anthropic" else OpenAIClient(model, provider)


def call(system: str, messages: list, model: str) -> dict:
    """One probe call, with reasoning at effort "high" on every provider so answers are comparable. What
    each provider is sent, besides the messages and max_tokens = max_output_for(model):

    - Anthropic, 4.6 generation and later (claude-opus-4-8, claude-sonnet-4-6, claude-opus-5, ...; also
      any claude-* id this module doesn't recognise): thinking {"type": "adaptive", "display":
      "summarized"} and output_config {"effort": "high"}.
    - Anthropic, older models (claude-opus-4-5-*, claude-haiku-4-5-*, ...): thinking {"type": "enabled",
      "budget_tokens": min(8000, max_tokens // 2)} and no effort setting.
    - OpenAI: reasoning_effort "high". Chat Completions returns no reasoning text, so thinking is empty.
    - OpenRouter: extra_body {"reasoning": {"effort": "high"}}; the reasoning text comes back in
      message.reasoning, which not every upstream model fills.

    No provider is sent temperature. If a model rejects the effort or thinking setting with a 400, the
    setting the error names is dropped (effort first if it names both) and the call retried; each setting
    is dropped at most once, the drops accumulate, and they are recorded.

    Returns {thinking, text, tool_use (None), usage, stop_reason, request}. stop_reason is "end",
    "max_tokens" (cut off at the output cap), "refusal", or the provider's raw value; an OpenAI refusal's
    text is returned as the text. request records the settings actually sent: {provider, thinking,
    effort, max_tokens, fallbacks}."""
    comp = make_client(model).complete(system, messages, max_output_for(model))
    return {
        "thinking": comp.thinking,
        "text": comp.text,
        "tool_use": None,
        "usage": comp.usage,
        "stop_reason": comp.stop_reason,
        "request": comp.request,
    }

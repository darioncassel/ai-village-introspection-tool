# AI Village Introspection Tool

Browse the [AI Village](https://theaidigest.org/village) logs as a timeline, find a specific moment, and
ask one of the agents about it. The tool rebuilds what that agent could see at that point and asks a
live model to answer in character, for example "what are you thinking about this?". It then puts the
answer next to what the agent actually did.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshot-dark.png">
  <img alt="Day 447 open in the timeline on the left; on the right, Claude Opus 4.8 is asked about Gemini 2.5 Pro's message just after it was posted" src="docs/screenshot-light.png">
</picture>

The AI Village is a long-running experiment by [AI Digest](https://theaidigest.org). Frontier agents
from many labs share chat rooms, pursue weekly goals set by human operators, and each drive their own
computer. The logs run from April 2025 to September 2026 (village days 1–535). This tool is independent of
AI Digest.

## System requirements

| | Requirement | Notes |
|---|---|---|
| **OS** | Linux, macOS or Windows | Developed and tested on Linux (Debian 13). The tool is pure Python with no compiled code of its own, so macOS and Windows should work, but they haven't been tested. The commands in this README are for bash or zsh. |
| **CPU** | Any 64-bit CPU. No GPU needed. | All model inference happens at the API providers. Loading the dataset uses one core for about 50 seconds on a 2.1 GHz Xeon; after that the server is mostly idle, waiting on API calls. |
| **Memory** | 4 GB free (an 8 GB machine is comfortable) | The server keeps the index in memory: about 3.2 GB, which is also the peak during loading. |
| **Disk** | About 1 GB | About 760 MB for the four dataset files and about 85 MB for the Python environment. The probe log grows by about 25 KB per question to an ordinary agent, and by up to about 0.5 MB per question to the Claude Code agent. |
| **Python** | 3.10 or newer | Tested on 3.10 and 3.13. |
| **Browser** | A recent Chrome, Edge, Firefox or Safari | Tested in Chromium. |
| **Network** | Outbound HTTPS | To huggingface.co once, for the download, and to your model provider for each question. The server itself listens only on localhost. |
| **Accounts** | A Hugging Face account with access to the dataset, and an API key for at least one of Anthropic, OpenAI or OpenRouter | The dataset is gated (see [Getting the data](#getting-the-data)). Browsing works without an API key; asking needs one. |

## Installation

```bash
git clone https://github.com/darioncassel/ai-village-introspection-tool.git
cd ai-village-introspection-tool
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e .
```

This installs the `village-introspect` command. The only dependencies are the `anthropic` and `openai`
Python SDKs.

## Getting the data

1. Request access at <https://huggingface.co/datasets/aidigestorg/ai-village>. Requests are reviewed
   manually, and you agree to AI Digest's research terms (see [Data terms](#data-terms-and-citation)).
2. Once you have access, download the four files the tool needs (about 800 MB):

   ```bash
   pip install -U huggingface_hub
   hf auth login
   hf download aidigestorg/ai-village \
       agents.jsonl.gz claude_code_messages.jsonl.gz events.jsonl.gz village-transcript.json \
       --repo-type dataset --local-dir ai-village-data
   ```

The rest of the dataset (screenshots, `computer_use_turns`, `agent_memories`, …) is not used.

## Setup

Everything is configured through environment variables:

| Variable | Required | What it does |
|---|---|---|
| `VILLAGE_DATASET` | yes | Directory holding the four dataset files. |
| `ANTHROPIC_API_KEY` | one key at least | Enables the Claude models. |
| `OPENAI_API_KEY` | | Enables OpenAI models such as `gpt-5.5`. |
| `OPENROUTER_API_KEY` | | Enables any `vendor/model` id through [OpenRouter](https://openrouter.ai). |
| `VILLAGE_STATE` | no | Where your probe log, stars and caches go. Default: `~/.village-introspect`. |
| `VILLAGE_MODELS` | no | Comma-separated model ids that replace the default list in the model picker. The first one becomes the default. |

For example:

```bash
export VILLAGE_DATASET="$PWD/ai-village-data"
export ANTHROPIC_API_KEY=...
```

The tool reads keys only from the environment. It doesn't load any `.env` file.

## Running

```bash
village-introspect                 # then open http://127.0.0.1:8765/
village-introspect --port 9000     # another port
```

`python -m village_introspect` works too. The command takes three options:

- `--port`: the port to serve on (default 8765).
- `--host`: the interface to bind (default `127.0.0.1`).
- `--workers`: how many model calls run at once (default 4).

A few things to know:

- **Warm-up.** The page shows "Loading the village…" for about a minute while the server indexes the
  events. After that, everything is instant apart from the live model calls.
- **First run.** The first run also works out each village day's time span from
  `village-transcript.json` and caches it under `VILLAGE_STATE`. A missing or corrupt transcript stops the
  load with a "Server error" in the page rather than showing an empty timeline.
- **Remote machine.** On a remote machine, forward the port instead of binding to a public interface:
  `ssh -L 8765:127.0.0.1:8765 you@host`. The server makes paid API calls with your keys and has no
  authentication, so keep it on localhost. It also refuses requests that don't address it as
  `localhost` or `127.0.0.1`.

## Using the viewer

The default view has **one pane: a vertical timeline of days**. A second pane (**Ask**) opens on the right
only when you ask a model something. Everything else is added on demand.

- **Timeline.** Every village day is a row, showing the date, message and agent counts, and a headline
  (the first operator post, or the most active agent).
  - The weekly **goals** appear as small labels between days. Goals detected automatically after mid-2026
    are tagged `auto`.
  - Days with no chat are hidden unless View ▾ → Activity events is on.
  - The lines on the left are the **chat rooms, drawn like git branches**. #general is the trunk. #best
    and #rest branch off when the village split into rooms (day 349) and merge back when everyone
    returned (day 447). #focus and a few small rooms are short side branches.
- **A day.** Click a day to open it; any other open day closes.
  - Its header stays pinned while you read, with ‹ previous · next › and ✕.
  - Inside, every message is already expanded. Very long ones get a "show more".
  - "Its logged thinking" is one click away: the agent's own thinking at that message, as its provider
    returned it. That is thinking blocks for Anthropic and Gemini models, a reasoning *summary* for
    OpenAI models, the reasoning text that models such as DeepSeek, Grok, GLM and Kimi return, or nothing.
  - "About this day" at the top summarises the goal announcement, who was around, who was talked about,
    the human posts, and any Claude Code windows.
- **Cross-posts are folded.** The same text posted in several rooms within 2 hours is shown once, as
  "also posted in #best 17:00, #rest 17:01". Click that to see the copies.
- **Ask.** Hover over or select a message to get **Ask ▸** and a **⋯** menu. The menu offers: ask someone
  else, focus, star, copy link and, for the Claude Code agent, its session. The Ask pane shows:
  - the agent and the moment. The default is **just after it wrote this**, so you're asking about the
    message itself. "Just before it wrote this", its decision point, is under ⚙ Options, along with the
    answering model, the context size and the prompt format (see [fidelity](#how-the-reconstruction-works-fidelity));
  - a **fidelity block** saying how good the reconstruction is (see below). The messages the model is
    given are marked with an accent line in the timeline;
  - a chat-style thread with the composer at the bottom.

  Answers run as server-side jobs, so you can keep browsing. Each answer offers:
  - "compare with what it actually did";
  - follow-up;
  - re-run;
  - "prompt used", the exact prompt that was sent;
  - a permalink (`#/probe/<id>`).

  **Resample** lets the model take the agent's next turn afresh, shown next to what the agent really did.
- **Asking someone else** right after a message asks how another agent reacts to it. An example is
  asking a Claude agent right after a struggling Gemini agent posts. The tool compares the answer with
  that agent's next message in the room. It warns, or tags the answer **COUNTERFACTUAL**, when that agent
  may not have been reading the room.
- **Search** (`/`) filters the timeline in place. It opens the densest day and folds non-matching
  messages into "N hidden — show" rows; `n` / `N` step through the hits across days. The grammar:
  - words are ANDed, and `"phrases"` are matched whole;
  - `from:<agent|family|operator>`, `about:<agent>` (approximate mentions), `room:`, `day:440-452`,
    `in:thinking` and `kind:`;
  - a bare `447`, `2026-06-22` or `ei:NNN` jumps there.
- **Focus** an agent (from the ⋯ menu or View ▾) to highlight its messages. A chip in the header offers
  "only these".
- **View ▾** adds activity events, an overview strip, room lanes on/off, newest first and compact rows.
  `?` lists the keyboard shortcuts.
- **Links.** The day and message you're looking at are kept in the URL, so a moment can be shared as a
  link with anyone running the tool. Links to your questions (`#/probe/<id>`) only open against your own
  probe log.

## How the reconstruction works (fidelity)

The village logs record *external* agents. Their exact input prompts are not in the dataset (the raw
`llm_calls` table isn't exported), so what the live model gets is a reconstruction. Every answer carries a
fidelity block that says how close it is.

- **NEAR-FAITHFUL: the Claude Code agent** ("Opus 4.5 (Claude Code)", about days 300–360). Its
  `claude_code_messages` stream keeps both sides of every turn, so its real conversation is replayed up
  to the chosen point.
  - Each of its chat messages maps to the tool call that posted it. "Open its Claude Code session" draws
    its real turns as a side branch, and clicking the gap between two turns asks it at that exact point.
  - The unit is a **window**: the stretch between two compactions or session resumes. A window that
    begins at a compaction opens with Claude Code's own compaction summary, which is in the dataset and is
    replayed.
  - Missing: the exact Claude Code system prompt (a flagged stand-in is used) and, for a window that
    begins where Claude Code resumed its session, the earlier context the agent still had, which is not
    replayed. The fidelity block lists this for each window.
  - Prior-turn thinking is shown for reading but not replayed, because it was ephemeral for the real
    agent.
- **APPROXIMATE: every other agent.** The model gets the room's recent chat, 80 messages by default, up to
  the anchor, plus a stand-in system prompt.
  - The chat excludes the chosen message for "just before" and includes it for "just after".
  - **Two prompt formats.** By default the asked agent's own messages are replayed as its earlier
    (assistant) turns, and everyone else's messages are user turns. That is closer to how the agent met its
    own words. It does leave out the computer-use steps between its messages, and when the answering model
    isn't the agent's own, it puts another model's words in its mouth. Untick ⚙ Options → "replay its own
    messages as its turns" to send the chat instead as a single message holding a labelled transcript
    ("Gemini 2.5 Pro [Google]: …"). The probe log records which format each question used.
  - The model does **not** get the agent's real system prompt, screen, memory, other rooms or earlier
    days.
  - "What it actually did" is the message itself when you ask the author just before it; the author's next
    message in the room when you ask just after it; and, when you ask a different agent, that agent's next
    message in the room within 2 hours.
  - A presence check flags an asked agent that wasn't in the room, up to COUNTERFACTUAL.
  - **Known caveat:** in a quiet room the 80-message window can reach back weeks, and much of it can be
    automated "pausing the village for today" lines. The fidelity block says how far back it reaches.
- **The model is told it's a reconstruction.** Both paths use a stand-in system prompt with a
  reconstruction note. The note says the conversation is rebuilt from the logs, what is missing and, for
  the turns format, that its "own" turns may have been written by a different model. The question itself
  says that nothing the model writes is shown to the other agents or appears in the village. It doesn't
  claim that no one reads it: every answer is logged.
- **The model voicing an agent can differ from the agent's real model**, and the fidelity block says so
  ("voiced by claude-opus-4-8"). With an OpenRouter key you can often voice an agent with its own model
  family (for example `google/gemini-3.1-pro-preview` for a Gemini agent).
- **An answer's thinking is usually a summary.** For Claude models it is requested as
  `display: "summarized"`, because Opus 4.8 otherwise returns empty thinking, so it is a summary written
  by the model, not raw chain of thought. OpenRouter models may return raw reasoning. OpenAI's API returns
  none.

## Probe models

The model picker lists these ids; the provider is inferred from the id:

| Model id | Provider | Note |
|---|---|---|
| `claude-opus-4-8` | Anthropic | default |
| `claude-opus-4-5-20251101` | Anthropic | era-matched for the Claude Code agent |
| `claude-sonnet-4-6` | Anthropic | |
| `claude-haiku-4-5-20251001` | Anthropic | fast |
| `gpt-5.5` | OpenAI | |
| `google/gemini-3.1-pro-preview` | OpenRouter | |
| `x-ai/grok-4.5` | OpenRouter | |
| `deepseek/deepseek-v4-pro` | OpenRouter | |
| `moonshotai/kimi-k3` | OpenRouter | |

- **Routing.** Ids of the form `vendor/model` go to OpenRouter, ids starting with `claude` go to
  Anthropic, and OpenAI's own ids (`gpt-*`, `o3`, `o4-mini`, `chatgpt-*`, `codex-*`) go to OpenAI. Other
  vendors' models need their OpenRouter id (`x-ai/grok-4.5`, not `grok-4.5`); a bare id like that is shown
  disabled in the picker, with a note saying so.
- **Reasoning.** Every provider is asked for high reasoning effort, and none is sent a temperature:
  - Claude 4.6 and later: adaptive thinking at effort high;
  - older Claude models such as Opus 4.5 and Haiku 4.5: a fixed thinking budget;
  - OpenAI: `reasoning_effort` high;
  - OpenRouter: `reasoning.effort` high.

  If a model rejects one of these settings, it is dropped and the call retried. Each probe records the
  settings actually sent, and any setting that was dropped.
- **Cut-off or refused answers.** An answer that hit the output limit, or that the model declined to
  give, carries a warning in the viewer and in the probe log.
- **Missing keys.** A model whose provider key isn't set is shown disabled, with a note naming the
  variable. If `claude-opus-4-8` isn't usable, the first usable model becomes the default.
- **Your own list.** Set `VILLAGE_MODELS` to use a different list, for example
  `VILLAGE_MODELS=claude-opus-4-8,google/gemini-3.1-pro-preview,moonshotai/kimi-k3`.

**Cost.** Every question is a live call billed to your key. Approximate probes send about 80 chat
messages. A near-faithful Claude Code probe replays that agent's whole context window up to the chosen
point, which can run to 100k+ input tokens per question.

## Your probe log

Everything you ask is kept locally in `VILLAGE_STATE` (default `~/.village-introspect`):

- `probes.jsonl`: every question, with the exact system prompt and messages sent, their sha256, the
  model settings used, the answer, and the fidelity block. It is append-only, and "prompt used" in the
  viewer reads from it. A damaged line (for example after a crash) is skipped with a warning at startup.
- `stars.jsonl`: starred messages and answers.
- `cache/`: the village day spans.

Nothing is sent anywhere except to the model provider you pick.

## Development

```bash
pip install -e ".[dev]"
pre-commit install    # run the hooks on every commit
pytest
```

The pre-commit hooks run ruff (lint and import order), black, basic file checks, nbstripout for
notebooks, and a gitleaks secret scan. `pre-commit run --all-files` runs them over the whole tree, except
gitleaks, which only checks staged changes.

The tests use synthetic corpora and fake API clients, so they need neither the dataset nor keys. Three
extra tests run against the real data:

```bash
VILLAGE_DATA_TESTS=1 VILLAGE_DATASET=... pytest
```

Code map (`village_introspect/`):

- `server.py`: a stdlib HTTP server and the JSON API. The index loads in the background, with progress
  on `/api/status`.
- `village_lib.py`: the timeline index covering days, goals, rooms, agents, mentions and search, plus
  probe routing and the fidelity text.
- `cc_lib.py`: the near-faithful Claude Code replay.
- `tierb_lib.py`: the approximate observable-chat reconstruction and the presence check.
- `probe_jobs.py`: the probe job pool, the probe log and stars.
- `llm.py`: the Anthropic, OpenAI and OpenRouter clients.
- `config.py`: the environment variables and the default model list.
- `viewer.html`: the whole front end in a single file.

## Data terms and citation

The AI Village dataset is published by AI Digest under their own research terms, which you accept when
you request access. See the [dataset page](https://huggingface.co/datasets/aidigestorg/ai-village) for the
current terms.

This repository ships none of the dataset files; you download them yourself. The screenshots, the weekly
goal titles and a few short quotes in the code come from the AI Village. Please cite the AI Village if you
use this tool in your work.

## Known issues

- The curated list of weekly goals misses some weeks and misdates a few, so some goal labels on the
  timeline are wrong.
- Claude Code windows that begin at a session resume don't include the context the agent resumed with
  (see [fidelity](#how-the-reconstruction-works-fidelity)).
- Thinking labelled as Anthropic or Gemini thinking blocks may itself be a provider-side summary for
  newer models.
- The viewer still has rough edges, for example around browser back/forward, keyboard access to some
  menus, and search hits in folded cross-posts.

## License

The code is MIT-licensed (see [LICENSE](LICENSE)). The dataset is covered by AI Digest's own terms.

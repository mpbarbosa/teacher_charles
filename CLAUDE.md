# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

Code, comments, docstrings, commits and docs in **Portuguese**. Everything the
student sees — persona, UI strings, model response — in **English**, no
exception: the product is "100% in English" by an audience decision
(`docs/plano-fase-1.md` §1.1).

## Commands

```bash
uv sync --extra dev                      # environment (Python 3.12, pinned)
python3 scripts/check_env.py             # RAM, disk, uv, Ollama, model, voice weights
uv run teacher-charles --reload          # http://localhost:8000
```

Voice (Phase 1.5) is an **extra and off by default** — text chat works in full
without it, and it is ~400 MB of wheels plus ~120 MB of weights:

```bash
uv sync --extra dev --extra voice
uv run python scripts/fetch_voice_models.py      # Kokoro; Whisper downloads itself
TC_VOICE_ENABLED=true uv run teacher-charles
```

```bash
uv run pytest                            # everything (integration included, hits the real Ollama)
uv run pytest -m "not integration"       # 522 tests, ~4s, no model at all
uv run pytest -m integration             # only the 9 that exercise the local Ollama
uv run pytest tests/test_turn.py::test_name -x
uv run ruff check . && uv run ruff format --check .
```

The voice tests follow the same rule: the stub ones run with no weights at all
(`test_voice*.py`), and the 8 that really load Whisper and Kokoro live under
`-m integration` and skip themselves when the weights are not on disk.

`uv run pytest` with no filter is **not** the fast path: the integration tests
skip themselves without Ollama, but with Ollama up they load a model and the
time goes from ~3s to minutes. During development use `-m "not integration"`.

To work on the interface without a model: `TC_PROVIDER=fake uv run teacher-charles`.
To use a model that is already downloaded: `TC_LLM_MODEL=qwen2.5:3b-instruct-q4_K_M uv run teacher-charles`.

## The principle that organizes the code

> **The pedagogical structure lives in the application, not in the prompt.
> A correction is structured data, not prose.**

Practical consequence: when there is a curated rule for the error, the
correction **cites the `rule_id`** from `data/contrastive/` instead of letting
the model invent grammar. That is why the `correction` table has existed since
the first migration, with a nullable `rule_id` — `NULL` marks "came from the
model alone", a distinction D4 measures and D7 shows. Before solving something
by adding a sentence to the prompt, check whether the right place isn't code in
`teaching/`.

## Layers

| | |
|---|---|
| `llm/` | Providers behind `base.ChatProvider`. Nothing outside this package knows whether the model is in Ollama, in a llama.cpp or in the cloud. Only `stream()` is abstract; `complete()` derives from it. |
| `teaching/` | The product. **Agnostic about where the text came from** — it takes a string and returns `Chunk`/`Analyzed`/`Leveled`/`Done`/`Failed` events. That is what allows STT/TTS to be attached in Phase 1.5 without rewriting the middle. |
| `voice/` | The two ends of Phase 1.5 — STT and TTS — behind `base.SpeechToText` / `base.TextToSpeech`, in the same arrangement as `llm/`. **It does not import `teaching/`**: it takes bytes and returns text; it takes text and returns PCM. |
| `store/` | SQLite through the stdlib's `sqlite3`. Schema in numbered `.sql` files, no ORM; the dataclasses in `models.py` only give the rows names. |
| `prompts.py` | Loads `prompts/*.md` with an mtime-keyed cache: editing the persona takes effect on the next turn, without restarting. |
| `app.py` | FastAPI. Assembles everything in the `lifespan` and keeps it in `app.state`. |
| `web/` | Single page: Jinja template + JS/CSS with no build step. |

`config.py` never imports `llm` **nor `voice`** — the dependency runs one way
only. That is why `default_options(settings)` lives in `llm/__init__.py` and not
in `Settings`.

## Invariants that are not obvious in the file you are editing

**One SQLite connection, shared.** `app.state.db` holds **one** connection with
`check_same_thread=False` and `isolation_level=None`. Hence the rule: **nothing
on a request's path opens a multi-statement transaction** — two requests can
interleave. Every repository method is a self-contained statement. The
migrations are the exception and run at startup, before the server accepts a
connection.

**The `BEGIN` goes inside the migration script.** `executescript` ends any
transaction already open, so a `conn.execute("BEGIN")` from outside would be
undone and the `ROLLBACK` would not protect partial DDL. See
`store/db.py::migrate`.

**The persona comes first and unaltered.** Ollama caches the prompt prefix;
interpolating a changing value into the system prompt throws that cache away and
the student pays for the re-evaluation every turn. A prompt that varies by level
goes in `prompts/partials/level/` and enters as a **second system message**, not
inside the persona: that way it only invalidates the cache when the level
actually changes. That is why the warm-up also sends the level partial of the
active profile.

**The mode changes policy, not just prose.** `teaching/modes.py` decides the cap
on corrections per turn (5 in `correction`, 2 in the others) and, in the drill,
**which rule** is trained — by querying the aggregated `correction`, never by
the model. One rule per session (`session.drill_rule_id`), not per turn: a drill
that swaps rules on every message is a quiz, and a stable block in the prompt is
a cached prefix.

**The order of the system messages is persona → level → mode → drill data**,
and then, at the end of the prompt, the curated hint and the drill request. The
beginning is ordered by stability (what changes least comes first, because of the
prefix cache); the end is ordered by effectiveness — measured twice, in D4 with
the hint and in D6 with the drill: an instruction far from the point of
generation gets diluted and the small model ignores it.

**The student's level is decided by code, not by the model.** `teaching/level.py`
has a deterministic rubric (sentence length, long words, connectives,
structures, fewer errors with a curated rule) calibrated in
`tests/fixtures/level_samples.yaml`. Touching the constants requires running
`scripts/measure_level.py` — the numbers are in §5.3 of the plan, and a test
breaks if accuracy drops. Two consequences that are not obvious in the file: the
rubric **does not estimate C2** (trying was off by two steps) and **no single
turn moves the level** (4 counted turns in a row, one step at a time). The
window closes on every level change, via `level_change.at_signal_id` — without
that the same four turns would move the level again on the next turn.

**The warm-up sends the turn's prefix, not a "hi" nor "just the persona".**
There is a single list, `TurnRunner.stable_prefix()`, and the warm-up and the
turn use the same one — because there were once two and they diverged: D5 and D6
added level and mode to the turn, the warm-up stayed on the persona, and the
first turn arrived with 53% of the prompt cached and 11.6s to the first word.
With the same list: 0.9s (plan §5.5). `_warm_up` never propagates an error —
`/health` already reports an unreachable model, and bringing startup down
because of the warm-up would be worse.

**No system message after the history.** The chat template joins the system
messages at the top, so a "hint at the end of the list" is pasted right after
the persona and invalidates the cache of the whole history — measured: the total
cached stayed stuck at the prefix's 674 tokens and the first word went from
2.8s to 9.1s over six turns. The curated hint and the drill request go **inside
the student's last message** (`TurnRunner._with_notes`). It is counterintuitive
enough for someone to "fix" it by accident; the test
`test_rule_hint_rides_with_the_student_message` exists for that.

**The student's utterance is recorded before calling the model**, and the
history already includes it — which is why `build_messages` does not add the new
message a second time.

**The microphone transcript goes to the text field, not to the turn.** STT makes
mistakes; an STT error that becomes a turn makes Charles correct an error the
student did not make — the same class of failure D3/D4 exist to avoid on the
model's side. `POST /api/stt` returns only the text and **records no turn at
all** (`test_nao_grava_turno_nenhum`). It is the only safeguard that does not
depend on which STT model is loaded, and it exists because accuracy with a
Brazilian accent is Phase 1.5's open debt (plan 1.5 §2.3).

**The audio does not travel over the SSE.** The `sentence` frames from
`/api/chat` carry an index and text; the page asks `/api/tts` for each sentence
in a request of its own. Base64 would be 33% more bytes on the text's critical
path — but the real reason is another: synthesizing inside the generator that
relays the chunks would delay the words to bring the sound forward. Kept apart,
sentence 1's synthesis runs while the model writes sentence 2, which is the only
reason per-sentence TTS exists.

**The index on the `sentence` frame is not decoration.** The requests to
`/api/tts` resolve out of order; the queue in `voice.js` plays by index. Playing
in arrival order would scramble the teacher's speech the first time the network
hesitated.

**A sentence boundary also counts at the end of a chunk**, and that is a latency
choice, not carelessness (`voice/segment.py`). Waiting for the space would cost
one token of delay on every sentence — ~130 ms at 7-8 tok/s — paid at the only
point in the turn where the student waits in silence. The one undecidable case,
a digit before the period (`3.` that may become `3.5`), waits; abbreviations and
initials are decided by what comes before the period.

**The voice engines load lazily and the warm-up swallows the error.** Bringing
startup down because of a missing weight would be worse than a degraded
`/health`. Hence `ready` and `last_error` in `voice/base.Loadable`: it is
`/health` that reports it, and `ready` never triggers a load — a health check
cannot have a side effect of hundreds of megabytes.

**`av` is pinned to `<19`.** `faster-whisper` 1.2's `decode_audio` calls
`av.open(metadata_errors=...)`, an argument PyAV 19 removed. With 19 installed,
**every** transcription blows up with a `TypeError` before the model sees a
sample. It is not version caution; it is a measured incompatibility.

**The active profile is `student.seen_seq`, a counter — not a timestamp.** Two
clock-based versions (second and millisecond) broke the same
switch-and-switch-back profile test: the writes land in the same instant and the
tiebreak always returns the same student.

**Provider errors are typed on purpose** (`ProviderUnavailable`,
`ModelNotFound`, `ProviderTimeout`, `ProviderBadResponse`). React by type;
never by a substring of the message.

**`/api/chat` is a POST with a JSON body and a response in SSE frames**
(`data: {...}`, types `chunk` / `done` / `error`). It is not `EventSource`: that
one only does GET, and the student's utterance does not go in a query string.
The `X-Accel-Buffering: no` header exists so a proxy on the path does not hold
the body and cancel out the streaming.

**`/health` answers 503 when degraded** (database or model down). It is
deliberate: separating "the service came up" from "it is ready to teach" avoids
a misleadingly green health check.

## Tests

`tests/conftest.py` erases every `TC_*` variable and clears the `get_settings`
cache in each test — a local `.env` has already broken a test here. Build
settings with `Settings(_env_file=None, **overrides)` (or the conftest's
`make_settings`); never rely on the machine's environment.

In `tests/test_level.py` the gate over the labeled samples is a **fraction**
(70% exact accuracy, zero two-step errors), not a fixed value: the suite can
grow without the test turning brittle, but a real regression breaks it.

In a route test, pass `llm_warmup=False`: the warm-up would consume one of
`FakeProvider`'s scripted responses and would show up in `provider.calls`,
messing up the assertions about what the turn sent to the model.

`FakeProvider` records every call (`calls`, `last_call`,
`system_prompt_of_last_call()`) and slices the response on purpose, so that
whoever consumes it has to deal with chunks as with the real provider.

The integration tests pick the model in order: `TC_TEST_MODEL`, the configured
`TC_LLM_MODEL` if it is downloaded, or the smallest one present on the machine —
nobody has to download 2.5 GB to get the suite green.

## The hardware budget decides the design

The minimum target is CPU/iGPU without CUDA, and official Ollama does not use
the Intel iGPU. Measured on this machine: **9.3 tok/s on a 3B Q4 and 4.0 on a
7B**, which puts a 4B at ~7–8 and an 8B at ~3.5 — which is why the default is
4B-class and not 14B. This is a project premise, not a configuration detail:

- Streaming is not an ornament: it is what makes 7–8 tok/s tolerable. Do not
  accumulate the response before relaying it.
- `DEFAULT_HISTORY_TURNS = 20` exists because evaluating the prompt costs real
  time (~54 tok/s on a 3B), not because of a context window limit.
- Reasoning models (Qwen3 among them) think before answering;
  `TC_LLM_THINK=false` turns that off. `None` lets the server's default stand.
- Python is pinned to `>=3.12,<3.13` on purpose: the `ctranslate2` and
  `onnxruntime` wheels, which Phase 1.5's voice will need, lag behind the new
  versions. Do not change the interpreter.

## Where the data lives

Three destinations, and confusing them is a category error:

- **Model weights** — in `~/.ollama`, never in the repo. The voice ones neither:
  Kokoro in `$XDG_CACHE_HOME/teacher-charles/voice` (a cache, re-downloadable)
  and Whisper in `huggingface_hub`'s cache. Phase 2's phoneme recognizer follows
  Whisper, for the same reason — it is a Hub repository, and the Hub already has
  a cache. **Never in `XDG_DATA_HOME`**, which is the student's personal data.
- **Student progress** — `$XDG_DATA_HOME/teacher-charles/charles.db`, outside
  the repo. It is personal data. `*.db` is in `.gitignore`.
- **Curated knowledge** — `data/contrastive/*.yaml`, **versioned**: it is
  product content, reviewed and tested item by item. Today `data/` is empty; the
  base arrives in D3.

## State, and where the code grows next

All of Phase 1's deliverables are done: D0 (foundation), D1 (provider), D2 (text
chat), D3 (contrastive base), D4 (structured correction), D5 (profile and
level), D6 (the three modes) and D7 (progress screen, README and latency). The
order and the definition of done for each deliverable are in
`docs/plano-fase-1.md` §5; `docs/analise-inicial.md` has the feasibility
analysis that gave rise to the project.

Phase 1 is complete, and so is **Phase 1.5 (voice)** —
`docs/plano-fase-1-5.md`. The student speaks via push-to-talk, Charles answers
out loud sentence by sentence, all local. `teaching/`, `llm/` and `store/` did
not change a single line for that: the voice came in as two ends, which was the
bet of the §4 detour.

Phase 1.5 left two debts, both measured in `docs/plano-fase-1-5.md`:
**5.4-7.4s to the first spoken word** against the 1-2s goal (and the largest
slice is the LLM, not the audio — the lever is the idle iGPU from §1.3), and the
**accuracy of the STT with a Brazilian accent, which was not measured** — the
audio in the tables is synthetic, and the 0% WER is a floor, not an expectation.

**Phase 2 is open** — `docs/plano-fase-2.md`: pronunciation by phonemes and
spaced repetition. It opened with measurement, as 1.5 did, and the measurement
has already moved the design in five places (its §3). The numbers that matter:

- Half of the pronunciation stack **is already installed** by the `voice` extra:
  the `phonemizer` + `espeakng_loader` that Kokoro brings give the G2P, and
  `onnxruntime` gives the runtime. What is missing is a 318 MB weight.
- The phoneme recognizer costs **RTF 0.17** — faster than the Whisper that is
  already on the path — and that is why it runs **after** the turn, never inside
  it: the path to the first word is already blown at 3.99s.
- The floor is **0.061** phoneme error rate on correct speech and the signal is
  **0.119**, with 6 of the 8 curated errors named by the right phoneme. But the
  Brazilian-accent proxy goes to **0.676**, and there the alignment collapses:
  when two out of every three phonemes disagree, nothing is *the* error.
- Hence D9 being a **gate and not a stage**: measuring with a Brazilian human
  voice is what decides whether the evaluator exists. No product before it.

What comes after is in `docs/plano-fase-2.md` §7: turn-taking without a button,
prosody, the analysis debt, and mobile/multi-user.

Two known debts from Phase 1, both measured and recorded in §5.5: the analysis
costs ~10s per turn with an error (the way out is a better model just for it,
and the measurement shows that does not disturb the conversation), and the first
word is at 2.7s against Phase 1.5's goal of 1–2s. §7 of the 1.5 plan closed the
second one by turning on the iGPU (0.9s) and did **not** close the first: the
analysis *generates* JSON, and generation ties between CPU and iGPU.

There are several student profiles on one machine, and the active one is the one
with the highest `seen_seq`; real login and multi-user are out of scope for
Phase 1 (plan §4).

Querying the contrastive base in Phase 1 is **lemma and pattern matching over
curated YAML**, not vector search: ~100 items fit in memory, the matching is
deterministic and debuggable (`"I have 30 years"` → `calque.age-have`, always),
and each item gets a test. Embeddings are noted as a Phase 2 item.

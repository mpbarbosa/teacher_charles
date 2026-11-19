# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

Code, comments, docstrings, commits and docs in **English**; **this file is
the exception and stays in English.** Everything the student sees — persona, UI
strings, model response — in **English**, no exception: the product is "100% in
English" by an audience decision (`docs/plano-fase-1.md` §1.1).

The deliverable labels are the ones the plans in `docs/` use — **E0–E13**, E for
*entrega* — so that a reference here and a reference there name the same thing.

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

The word cards on a lesson's full text need a lookup file built once from
WordNet (~40 MB download, 34 MB file, about a minute). `wn` is needed only for
that build, so it is passed with `--with` and is in no extra:

```bash
uv run --with wn==1.1.1 python scripts/fetch_dictionary.py
```

The Structure button (the syntax view) is an **extra and off by default** too,
for the voice's reason — ~267 MB of wheels, spaCy and its small pipeline:

```bash
uv sync --extra dev --extra syntax
TC_SYNTAX_ENABLED=true uv run teacher-charles    # or scripts/run.sh --syntax
```

```bash
uv run pytest                            # everything (integration included, hits the real Ollama)
uv run pytest -m "not integration"       # ~1070 tests, ~40s, no model at all
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

E13 added the second half of that: **what the application knows about the
student is structured data too, not a ranking.** The drill queue stopped being
an error count and became a review schedule, because a count only knows how to
add up — it never records that the student learned.

Practical consequence: when there is a curated rule for the error, the
correction **cites the `rule_id`** from `data/contrastive/` instead of letting
the model invent grammar. That is why the `correction` table has existed since
the first migration, with a nullable `rule_id` — `NULL` marks "came from the
model alone", a distinction E4 measures and E7 shows. Before solving something
by adding a sentence to the prompt, check whether the right place isn't code in
`teaching/`.

## Layers

| | |
|---|---|
| `llm/` | Providers behind `base.ChatProvider`. Nothing outside this package knows whether the model is in Ollama, in a llama.cpp or in the cloud. Only `stream()` is abstract; `complete()` derives from it. |
| `teaching/` | The product. **Agnostic about where the text came from** — it takes a string and returns `Chunk`/`Analyzed`/`Leveled`/`Done`/`Failed` events. That is what allows STT/TTS to be attached in Phase 1.5 without rewriting the middle. |
| `voice/` | The two ends of Phase 1.5 — STT and TTS — behind `base.SpeechToText` / `base.TextToSpeech`, in the same arrangement as `llm/`. **It does not import `teaching/`**: it takes bytes and returns text; it takes text and returns PCM. |
| `store/` | SQLite through the stdlib's `sqlite3`. Schema in numbered `.sql` files, no ORM; the dataclasses in `models.py` only give the rows names. |
| `teaching/phonetics.py` | The G2P behind the IPA line that runs under every message. Text in, text out — no audio anywhere, which is why it is here and not in `voice/`. Lazy, and a failure costs the line, never the turn. |
| `teaching/syntax.py` | The parser behind the Structure button: text in, a sentence's parts out. Off the turn's path entirely — it runs from `POST /api/syntax`, when the student presses the button, and yields no `TurnEvent`. `analyse` reads tokens by attribute and imports nothing from spaCy, so its tests run with the extra absent. |
| `teaching/schedule.py` | The review schedule (E13). It knows **when** a rule comes back and knows nothing about the contrastive base; the `DrillPlanner` knows **which** rules can become a drill. Neither of the two decides on its own. |
| `teaching/dictionary.py` | The word card under the pointer on a lesson's full text: a word in, its senses out, each with an English definition, an example and the Portuguese for it. Read with stdlib `sqlite3` from a file `scripts/fetch_dictionary.py` builds out of Open English WordNet 2024 and OpenWN-PT, joined by interlingual sense. No `wn` at runtime, nothing loaded until a hover, and it records nothing. |
| `teaching/lessons.py` | Curated lessons: an authentic text analysed at six levels, one YAML per lesson in `data/lessons/`. The source is **linked, never copied** — `Source` has no field for the text, and `Point.cites` is held to 5 words per citation and a tenth of the source in total, both checked at load. A machine that holds its own copy in `corpus/lessons/<slug>.txt` gets it shown whole on the page — read per request, never in `/api/lessons`. Charles never sees a lesson; the page shows it, and the IPA in it comes from espeak at serve time. |
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
**which rule** is trained — by querying the database, never by
the model. One rule per session (`session.drill_rule_id`), not per turn: a drill
that swaps rules on every message is a quiz, and a stable block in the prompt is
a cached prefix.

**The drill queue is a schedule, not a ranking.** What orders it is
`Scheduler.queue` (E13): overdue first, with the frequency in `correction` left
as the *last* tiebreak between equally urgent rules. The swap was not cosmetic —
frequency is a number that only goes up, so the most-missed rule in the history
went on being the most-missed one after the student had got it right ten times,
and the drill never let it go. Reverting the sort key to `-times` leaves four
tests red, one of them from E6.

**Getting it right is the absence of an error where the error would have shown
up, and that is asymmetric.** The analyzer returns errors, not successes, so a
success can only be inferred — and it only counts inside the drill, over the
rule being trained, because there the exercise built the opportunity. A
correction by the rule is an error in any mode; a clean turn in free
conversation **is nothing at all**, and counting it would graduate rules by
silence. The guard that matters most: **an analysis that did not run graduates
nobody** — without it, an analyzer that is down would turn into a success,
because it produces the same absence of corrections. A known blind spot, noted
in §6 of the plan: inside the drill, "did not err" and "did not try" are the
same observation.

**And that guard is two conditions, because `Analyzer` never returns `None`.**
It was written as `analysis is None`, which covers only an absent analyzer: a
model that times out or a provider that dies gives back
`Analysis(degraded=True)` — zero findings, the same shape as a clean turn. So
the sentence above was false for exactly the case it was written for, and it
was measured at its worst: the student writes the drill's own calque, the base
matches it, the analysis model falls over, and the review came out `hit=True`,
crediting the rule to the student who had just missed it. Three of those
graduate it. The condition is now `analysis is None or not
analysis.is_evidence`, and that property is where the reasoning lives —
including why `skipped` is deliberately *not* part of it: there the base ran
and matched nothing, which is a poor observation rather than no observation,
and it is the blind spot the paragraph above already names. Narrowing
`is_evidence` to exclude `skipped` as well turns
`test_uma_resposta_vazia_de_conteudo_conta_como_acerto` red, which is how the
scope is held in place.

**The first rung of the interval ladder is worth half a day, not zero**, and
that is what replaces E6's rotation. With zero, a rule just missed falls due at
once and goes back to being the most overdue one in the next session, forever —
the student would never see the second rule. `DRILL_ROTATION` survived only for
the case the schedule cannot see (a rule picked in a session that ended with no
answer), and as a sort term instead of a filter.

**The two writes of a review are not one transaction.** That is the
shared-connection rule, and the order is chosen because of it: `rule_review`
(the event) before `rule_schedule` (the state), because the state can be
rebuilt from the event and not the other way round.

**A correction the verifier barred is kept, not just counted.** `_Verifier`
discards every finding that cites no curated rule — that is E4's guard against
the small model inventing grammar, and it has not changed. What changed is that
the span and the model's proposal now land in `unverified_finding` instead of
in `log.debug`: each row is an error in a real student's text that
`data/contrastive/` does not cover, which is the curation backlog, read
aggregated by `scripts/curation_queue.py`. Two things hold it in place. The
write goes **after** the corrections, same shared-connection reason as the
review pair: these are loose statements, and losing a curation row costs a
note the next similar turn offers again, while losing a correction costs the
lesson. And `Analysis.as_dict` **does not expose `rejected`** — that payload
feeds the SSE, and those findings are exactly the improvised grammar the
verifier just refused; material for the curator is not material for the
student. Both are mutation-tested in `tests/test_curation_queue.py`.

The queue has a blind spot that is written into the migration because it is
easier to rediscover than to deduce: with `require_curated_rule` on, the
analysis does not run when the base raises no candidate, so **every row comes
from a turn where the base already matched some other rule**. The deeper hole —
an error the base cannot see at all — is invisible by construction and cannot
be closed with data: without calling the model, "correct English" and "an error
the base misses" produce the same absence of candidates, which is §6's
asymmetry again. `TC_REQUIRE_CURATED_RULE=false` turns the queue into the
opposite probe, at the latency the default protects; that is a deliberate
curation run, not normal use.

**The order of the system messages is persona → app capabilities → level →
mode → drill data**, and then, at the end of the prompt, the curated hint and
the drill request. The beginning is ordered by stability (what changes least
comes first, because of the prefix cache); the end is ordered by effectiveness —
measured twice, in E4 with the hint and in E6 with the drill: an instruction far
from the point of generation gets diluted and the small model ignores it.

**Charles knows what the app can do, and that knowledge is a partial, not the
persona.** `prompts/partials/app/voice.md` and `.../text.md`, picked by
`teaching/capabilities.py::partial_name(settings.voice_enabled)`. It is second
in the prefix because it is the most stable thing left — voice is decided at
startup and does not change while the server is up, while the level changes
every many turns and the mode the student switches on screen. It exists because
without it the model answered from its generic chatbot priors: asked whether he
had a microphone to capture the student's voice, Charles said he did not and
suggested recording elsewhere — with the push-to-talk button on screen beside
that very sentence. `TurnRunner` takes a `bool`, not the `Voice` object:
`teaching/` does not import `voice/`, and that holds here too.

**He does not invent facts about sounds either**, which is the sibling the
persona's "do not invent grammar rules" had been missing. Asked for the IPA, he
offered *pat* against *bat* as a vowel contrast — the vowel is the same in both,
and the contrast is the initial consonant he had himself proposed a turn
earlier. Phase 2's phoneme evaluator does not exist (E9 is an unpassed gate),
and what reaches him is the transcript and never the sound, so **neither voice
state lets him claim to hear or score pronunciation.**

**The IPA on screen comes from espeak, never from the model** — same rule as
grammar, one layer down. `teaching/phonetics.py` runs the `phonemizer` that
`kokoro-onnx` already brings, so the symbol the student reads is the symbol the
voice pronounces; a second G2P would have the screen teaching against the
loudspeaker. It lives in `teaching/` and its two wheels are **base
dependencies**, not part of the `voice` extra, because nothing here is audio:
text goes in, text comes out, and hanging it off the extra would make the line
vanish in the install everybody does — voice is off by default. The partial
`prompts/partials/app/phonetics.md` tells Charles the line is there so he stops
typing symbols himself, and it rides inside the app-capabilities message rather
than adding one: both are truths about the product, fixed at startup, so the
prefix keeps its message count. The transcription is still of the **written
words** — nobody heard the student — and E9's gate is what would change that.

**One IPA line per bubble, not per sentence**, which is the opposite of the TTS
next to it. The segmenter exists so synthesis of sentence 1 runs while the model
writes sentence 2; here there is nothing to bring forward, because the G2P
finished before the frame reached the page. Per sentence it would only produce a
patched line where the last word of one and the first of the next never meet.

**The syntax breakdown comes from a parser, never from the model** — the same
rule a third time, after grammar and sounds. `teaching/syntax.py` runs spaCy's
small pipeline and every word the student reads on the panel is chosen in its
closed tables (`PARTS_OF_SPEECH`, `WORD_ROLES`); a tag outside them shows as
`other (tag)` instead of being guessed. Two things are left out on purpose.
Person, number and tense: the pipeline tagged *was* in *I was tired* as third
person, and there is no telling which of its morphology calls are the wrong
ones. And meaning: *in a sewer* is "prepositional phrase", never "place".

**A breakdown is not a verdict, and `NOTE` is what says so.** *I didn't went to
the party* parses as cleanly as a correct sentence — the structure is fine, the
mistake is in the verb form — so a tidy panel under a wrong sentence would read
as approval. The note travels in the payload (`Breakdown.as_dict`) and not as a
string in the template, so a test can hold it there. §6's asymmetry again: the
absence of a complaint is not a pass.

**The button is on screen only when asked for *and* installed**, and one
function decides it for the page and for the prompt (`syntax.on_screen`).
Asked for without the extra, the lesson is whole, the button is absent, and
`/health` carries the reason without changing the status. `installed` asks the
import system (`find_spec`) and imports nothing: importing spaCy costs ~3 s,
which is also why the warm-up runs in a thread, unlike the IPA's.

**Telling Charles about the button at the top of the prompt made him worse,
and the partial is short because of it.** Measured on qwen2.5 3B, three runs
each way, asked for a syntax analysis: with no partial he answered in two
sentences of prose; with a partial that described the panel ("subject, verb,
objects… the part of speech of each word") he typed a breakdown three times out
of three — *Verb phrase: am in*, *Present Continuous Tense*. The description
primed the thing it forbade. What works is what E4 and E6 already measured:
the instruction goes **inside the student's last message**
(`syntax.request_note`, through `_with_notes`), only on a turn where a pattern
says the student asked, and then he pointed at the button six times out of
six. `partials/app/syntax.md` names the button and nothing else, and a test
keeps the priming words out of it.

Both 4B builds were measured afterwards, same question, and both agree with
what ships. With the note and the short partial, the default
`qwen3:4b-instruct` pointed at the button in all eight answers it gave and
analysed in none (a ninth run timed out with no answer); bare `qwen3:4b` — the
default when the button shipped, the thinking build — did it nine times out of
nine.

**On the default the note is not a refinement, it is the feature.** With no
syntax text at all, `qwen3:4b-instruct` typed its own word-by-word analysis
three times out of three, which neither the 3B nor the thinking build did. The
descriptive partial did not stop it — two of three still analysed — and the
short partial alone did not either: one pointed at the button, one analysed,
and one invented a correction (*"we don't usually say I'm in a sewer"*) for a
sentence with nothing wrong in it.

The three models disagree only about the descriptive partial: it primed the
3B, failed to restrain the instruct build, and was obeyed by the thinking
build, which rereads the prohibition. Three runs a side is still a small
sample.

**`/api/syntax` records nothing** — no turn, no correction, no review. Looking
at a sentence is not evidence, for the reason `/api/stt` records no turn.

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
turn use the same one — because there were once two and they diverged: E5 and E6
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
student did not make — the same class of failure E3/E4 exist to avoid on the
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

**`think: false` is not sent on trust.** In Ollama the field does two things,
and only one of them is up to the model: it is offered to the chat template,
which may use it to tell the model not to reason, and it stops the server from
splitting reasoning out of the reply into `message.thinking`. A template that
never reads the flag leaves the model reasoning exactly as before, with nothing
separating it any more. That is bare `qwen3:4b`, the default until it was
measured: the tag is Qwen3-4B-Thinking-2507, whose template opens `<think>`
whatever the request says. Measured on Ollama 0.35, one-word answer: 179 tokens with `think: false`,
247 with `true`, 280 with the field absent — so `false` saved nothing, and it
put the whole chain of thought, a bare `</think>` and then the answer into the
reply the student reads. Hence `OllamaProvider._think_off_is_obeyed`: one
`/api/show` per provider, and `false` goes on the wire only when the template
reads `.Think`. For every other model the field is left out, which is the one
setting under which the server hides the reasoning. The word boundary in that
check is not pedantry — `.Thinking` and `.IsThinkSet` are in every thinking
template, the leaking one included, and neither is the flag. Unknown (lookup
failed, or a template that is not a Go template at all) lands on the side that
cannot leak — which is where the default sits: `qwen3:4b-instruct` shows a
Jinja template with no `.Think` in it, so the field is never sent to it, and
measured it gives the same two-token answer either way.

**The warning reads the template, not the capability list.** It says reasoning
cannot be turned off, and the only thing `/api/show` offers that proves it is a
template that follows the assistant header with `<think>` by itself.
`qwen3:4b-instruct` lists `thinking` among its capabilities and does not reason;
keyed on that list, the warning was false at every startup of the default
model.

Two things follow that are easy to undo. The analyzer and the warm-up still
pass `think=False` on every call, and that is right: it is a request, and the
provider is the layer that knows whether this model can honour it. And **hiding
the reasoning is not turning it off** — on a thinking-only model every turn
still waits for it, and the analysis spends its 320-token ceiling thinking
before it reaches the JSON. That is a property of the model and no flag fixes it; the
lever is a model that does not reason. The second line of defence is in
`extract_json_object`, which drops reasoning that closes a tag it never opened:
a model thinking about the analysis rehearses the JSON it was asked for, so the
first `{` of an unseparated reply is inside the thought.

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
7B** — which is why the default is 4B-class and not 14B. This is a project
premise, not a configuration detail:

- Streaming is not an ornament: it is what makes 7–8 tok/s tolerable. Do not
  accumulate the response before relaying it.
- `DEFAULT_HISTORY_TURNS = 20` exists because evaluating the prompt costs real
  time (~54 tok/s on a 3B), not because of a context window limit.
- Reasoning models (Qwen3 among them) think before answering, and
  `TC_LLM_THINK=false` turns that off **only on a model whose chat template
  reads the flag**. Bare `qwen3:4b` is not one of them — see "`think: false`
  is not sent on trust" above — and the default `qwen3:4b-instruct` does not
  reason, so there the flag has nothing to turn off. `None` lets the server's
  default stand.
- Python is pinned to `>=3.12,<3.13` on purpose: the `ctranslate2` and
  `onnxruntime` wheels, which Phase 1.5's voice will need, lag behind the new
  versions. Do not change the interpreter.

**The 4B figure was an interpolation until it was measured, and it came in
low.** The plans put a 4B at ~7–8 tok/s by size ratio (~80% of the 3B). Measured
side by side on the same day (`docs/plano-fase-1.md` §5.5), `qwen3:4b-instruct`
generates at **~55% of the 3B's rate**. Every "7–8 tok/s" above and in the plans
is that estimate, not a measurement. The same table says the 3B ties the 4B on
structured correction (27 vs 28 of 31, no false positive in either) and beats it
on every latency line — so the 4B default rests on decision 1.3's bet about the
teacher's prose, which no script here measures. That is an open debt, and if
the prose ties, the 3B is the better default.

**The default is `qwen3:4b-instruct`, and the suffix is load-bearing.** Bare
`qwen3:4b` is the `Thinking-2507` build: its chat template ends every prompt
with an unconditional `<think>` and never reads the request's `think` field. It
was the default from E0 while every number in the plans came from a 3B or a
7B; measured at last, its first word took 231s at the median (479–1124 tokens
of reasoning per turn) and all 31 analysis calls hit the 45s deadline with no
JSON. Before "fixing" a model name by dropping the suffix, read `/api/show`'s
`general.finetune`.

## Where the data lives

Five destinations, and confusing them is a category error:

- **Model weights** — in `~/.ollama`, never in the repo. The voice ones neither:
  Kokoro in `$XDG_CACHE_HOME/teacher-charles/voice` (a cache, re-downloadable)
  and Whisper in `huggingface_hub`'s cache. Phase 2's phoneme recognizer follows
  Whisper, for the same reason — it is a Hub repository, and the Hub already has
  a cache. **Never in `XDG_DATA_HOME`**, which is the student's personal data.
- **Student progress** — `$XDG_DATA_HOME/teacher-charles/charles.db`, outside
  the repo. It is personal data. `*.db` is in `.gitignore`.
- **The syntax pipeline** — in the virtualenv, as a wheel (`en_core_web_sm`,
  from spaCy's GitHub releases, pinned by URL in the `syntax` extra). spaCy
  ships its models as packages, so there is no cache to choose and nothing to
  fetch after `uv sync`. Bump it together with spaCy's minor version, never
  alone: a pipeline only loads under the minor it was trained for.
- **The word cards' lexicon** — `$XDG_CACHE_HOME/teacher-charles/dictionary/`,
  a cache like Kokoro's: built from two public lexicons by
  `scripts/fetch_dictionary.py`, re-downloadable, nobody's personal data. Not
  in `data/` either: WordNet is somebody else's work, and the card credits it.
- **Curated knowledge** — `data/contrastive/*.yaml`, and with it `data/vocab/`
  and `data/lessons/`, **versioned**: it is
  product content, reviewed and tested item by item. Today `data/` is empty; the
  base arrives in E3.
- **Third-party material** — `corpus/`, outside version control, `corpus/*` in
  `.gitignore` with the README as the one exception. Subtitles, transcripts,
  excerpts: authentic English somebody else owns, and this repo is public, so a
  commit here would publish it. The asymmetry with the line above is the whole
  point — a derived observation (*this construction appears 12 times in 30
  minutes of dialogue*) is ours to version; the 30 minutes are not. **Derive
  into `data/`, never copy into it.** `corpus/README.md` records what each path
  held, so a script that reads one can be understood with the file absent.

## The checkout is shared, and the register of it is not versioned

Several agents and the user work in this checkout at the same time. `HEAD`, the
index and the working tree are therefore **not yours** — any of them can move
between two of your own commands. The failure that follows is not a merge
conflict: it is a `git commit` that exits 0, prints a plausible summary, and
records something else — on a branch you never chose, or carrying paths another
session staged minutes ago.

`.claude/worktrees/COORDINATION.md` is the register of worktrees in use. The
`.gitignore` keeps it out of the repo and it lives only on the machine, so **a
fresh clone does not have one** and nothing in the tree points at it: this file
is the only place it can be discovered from. If it is absent, create it before
the first `git worktree add` — a `# Worktrees em uso` heading, the obligations
below, and a table whose columns are worktree path, branch, base commit, session
and what it is for.

Three obligations: whoever creates a worktree writes its row, whoever adopts
someone else's rewrites that row with their own branch, and whoever removes one
deletes the row. Write the row in the same breath as the `git worktree add` — a
worktree that exists with no row is the state the register exists to make
impossible.

**But a missing row is never licence to delete anything.** The register is kept
by hand, so it is evidence and not truth: an interrupted setup leaves a real
worktree unregistered, and a clean, quiet directory with no row is
indistinguishable from an already-merged one by any command. Reconcile against
`git worktree list` before sweeping. Worktrees have been swept away for want of
that check — and a session has already created one here without reading the
register at all, leaving a second session to write the row and ask the creator
to come back for it.

Two habits follow, and both are cheaper than the recovery:

- **Prefer your own worktree to the shared checkout.** `git worktree add` gives
  you your own `HEAD` and your own index, and none of the above can reach you.
- **Never `git stash`.** The stack is shared across the root checkout and every
  worktree, so what you pop may be another session's in-flight edit, which from
  their side looks like a file reverting by itself. To set work aside, commit it
  **with an explicit pathspec** — `git commit -m "…" -- PATH` — because a bare
  `git commit` takes the whole shared index, other sessions' staged paths
  included. That is the trap of the first paragraph, one command later.

## State, and where the code grows next

All of Phase 1's deliverables are done: E0 (foundation), E1 (provider), E2 (text
chat), E3 (contrastive base), E4 (structured correction), E5 (profile and
level), E6 (the three modes) and E7 (progress screen, README and latency). The
order and the definition of done for each deliverable are in
`docs/plano-fase-1.md` §5; `docs/analise-inicial.md` has the feasibility
analysis that gave rise to the project.

**E13 (spaced repetition) is delivered**, out of order and on purpose: §5 of the
Phase 2 plan says it does not depend on E9's gate, and it was the path taken
while E9 stays open. What it swapped out is in §7 of `docs/plano-fase-2.md`, and
the two debts it opened are in §6 there.

**The IPA line is delivered too, and it has no deliverable number**: it was not
in any plan, so numbering it would take a label the Phase 2 author may want for
something else. §5 of `docs/plano-fase-2.md` records it under the table. It
does not depend on E9 either, and for the reason the gate exists: the line says
how the written word is pronounced, which is a table; E11 would judge how the
student said it, which is measurement over audio.

**The syntax view has no number either** — the Structure button under each
message, `teaching/syntax.py` and `POST /api/syntax`. It was in no plan, and it
does not depend on E9 for the IPA line's reason: it reads the written words.

**The word cards have no number either** — resting the pointer on a word of
a lesson's full text opens its senses, English definition and Portuguese, from
`teaching/dictionary.py` through `POST /api/define`. **The gloss comes from a
lexicon, never from the model** — the same rule a fourth time. Two things the
card owns up to, because they are easy to "fix" wrongly: it shows senses in
WordNet's order and does not know which one the text means (`NOTE` travels in
the payload, for `syntax.NOTE`'s reason — picking the sense is disambiguation,
a model's job with a model's error rate); and a word written in lower case
drops every capitalised lemma, because running text never writes a name or a
symbol in lower case — without that, *as* opened on American Samoa and arsenic.
It is on by default and shows only when the lookup file exists; without it the
words are plain text and `/health` says how to build it.

**Lessons have no number either** — `/lessons`, `teaching/lessons.py` and
`data/lessons/`. A lesson is a reading of somebody else's text, so it links to
the source and versions only the analysis; the loader refuses a lesson whose
citations grow past the budget. It stays out of the prompt, so it costs the
prefix cache nothing. §5 of `docs/plano-fase-2.md` records it.

**The curation queue has no number either**, and it is recorded in the same
place for the same reason. It keeps the correction the verifier barred instead
of only counting it, so that what the contrastive base fails to cover becomes a
list a curator can read and not just a number — `unverified_finding` and
`scripts/curation_queue.py`. §5 of `docs/plano-fase-2.md` states its blind
spot, which the invariants above also carry.

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
- Hence E9 being a **gate and not a stage**: measuring with a Brazilian human
  voice is what decides whether the evaluator exists. No product before it.

What comes after is in `docs/plano-fase-2.md` §8: turn-taking without a button,
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

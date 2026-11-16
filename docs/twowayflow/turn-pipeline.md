# One turn, as data

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Name         turn-pipeline                                                                  │
│ Type         Dataflow graph — twowayflow `*.flow.ts`                                        │
│ Project      teacher_charles                                                                │
│ Description  One turn of POST /api/chat as data dependencies: the student's utterance to    │
│              the SSE frames and the rows SQLite keeps, with the review cycle that turns     │
│              today's error into a later session's drill.                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

The diagram is [`turn-pipeline.flow.ts`](turn-pipeline.flow.ts), beside this file. It is not
reproduced here, and that is twowayflow's own rule rather than an omission: *code on disk is the
only source of truth, and the canvas is an editable projection over its syntax tree.* A second
copy in a Mermaid block would be the sidecar model that architecture exists to avoid — it would
drift, and nothing would say which half was wrong.

To look at it: open a [twowayflow](https://github.com/mpbarbosa/twowayflow) checkout in VS Code,
press <kbd>F5</kbd> ("Run twowayflow extension"), and in the new window open this `.flow.ts` and
run **twowayflow: Open Flow Canvas**. Or install the extension into your own editor with that
repository's `scripts/twf-install` and skip the <kbd>F5</kbd>.

Read without the canvas it is still a flat list: every node on one line, every edge on one line,
the prose for each box in its `description`.

## What it says, in one paragraph

30 nodes, 58 edges. Five sources (the prompt files, the curated base, the student's utterance, the
profile-and-session row, the microphone), seventeen transforms, one fan-in (`buildMessages`), one
filter (`_Verifier`), six sinks — the SSE frames, the audio queue, and four nodes covering the six
analysis-and-schedule tables a turn writes. `turn` itself is written by `appendStudent` and
`appendTeacher`, which are transforms rather than sinks because the `turn_id` they return is what
every later row hangs off. The spine is `studentText → appendStudent → buildMessages → stream → sseFrames`. Below
it the analysis runs *after* the reply has streamed, because Ollama serves one request at a time
on this machine and firing both together only delayed the first word. Around the whole thing runs
one cycle — `observeSchedule → scheduleRows → queue → sessionRow → drillTarget → buildMessages` —
which is E13: the rule the student got wrong today is chosen at the *next* `POST /api/mode`, not
in this turn.

## What each claim rests on

Pinned at `96e0249`, with `uv run pytest -m "not integration"` reporting 852 passed, 13 skipped.

| Claim in the diagram | Held by |
|---|---|
| `appendStudent` happens before `stream` — a failed generation keeps the utterance | *"test_provider_failure_keeps_student_text"* |
| `analyze` waits for the last chunk by default | *"test_analysis_waits_for_the_last_chunk_by_default"* |
| `verify → sseFrames` lands before the `done` frame | *"test_analysis_event_comes_before_done"*, *"test_chat_emits_the_analysis_before_done"* |
| `ruleHint → buildMessages` enters *inside* the student's last message, not as a system message | *"test_rule_hint_rides_with_the_student_message"* |
| `verify → sseFrames` carries findings only — never the rejected ones | *"test_the_rejected_finding_never_reaches_the_student_payload"* |
| `verify → gaps` keeps the barred finding instead of only counting it | *"test_the_rejected_finding_is_kept_and_not_just_counted"*, *"test_the_turn_files_the_rejected_finding_in_the_queue"* |
| a failed `gaps` write does not cost the lesson | *"test_a_broken_queue_write_does_not_cost_the_lesson"* |
| no `verify → observeSchedule` edge when there is **no analyzer at all** — nothing graduates | *"test_sem_analise_nao_ha_revisao"* |
| a **degraded** analysis graduates nobody either — the guard is two conditions, not `None` alone | *"test_a_degraded_analysis_graduates_nobody"*, and the three mutations below that turn it and the blind-spot test red |
| a correction citing no curated rule does not reach the schedule | *"test_correcao_sem_regra_curada_nao_entra_na_agenda"* |
| `queue` is ordered by the schedule, with the error count as the last tiebreak | *"test_a_fila_e_a_agenda_e_nao_a_contagem"* |
| the ladder's first rung is half a day, and a lapse lands there rather than at zero | *"test_a_escada_mede_os_intervalos_em_dias"*, *"test_o_lapso_cai_no_degrau_de_reaprendizado_e_nao_no_zero"* |
| three consecutive hits graduate a rule; one does not | *"test_tres_acertos_seguidos_graduam_a_regra"*, *"test_um_acerto_isolado_nao_gradua"* |
| `observeLevel` needs four agreeing turns — one turn never moves the level | *"test_one_turn_never_moves_the_level"* |
| the level enters as a system message of its own, after the capabilities | *"test_level_prompt_is_a_system_message_of_its_own"*, *"test_the_capabilities_are_the_second_system_message"* |
| `spellStudent → sseFrames` arrives before the teacher's first chunk | *"test_a_fonetica_do_aluno_vem_antes_do_primeiro_pedaco"* |
| `stt → studentText` fills the field and records no turn | *"test_nao_grava_turno_nenhum"* |
| `segment → sseFrames` emits one indexed frame per closed sentence | *"test_cada_sentenca_fechada_rende_um_quadro_indexado"* |
| `contrastiveBase` holds 44 curated items in 3 files | the probe below |
| the projection parses with no diagnostics and no unparsable links | the probe below |

Each row cites a test that exists in `tests/` at this revision — resolved mechanically, not by
memory. None of those tests has been watched go red *for this document*; where a guard's own
mutation test matters most (`verify → gaps`, and the absence of `rejected` from the student
payload), `tests/test_curation_queue.py` already does that, and
[`mutation-test-guards`](../../CLAUDE.md) is the discipline the rest of the suite is held to.

The resolver treats straight quotes inside italics as a cited test name and looks it up, so prose
quotations in this file use typographic quotes instead. Two sentences quoted in the revisions
table at the end were reported as missing tests until they were re-marked — the checker being
right about its own convention and wrong about the document. Writing that convention out in this
paragraph in its own notation made the paragraph fail too, which is the shortest demonstration of
the point available.

And a claims table says nothing about the claims it does not make. The row below it — the
degraded analysis — is in this table *because* drawing it showed the row above was narrower than
the sentence it was holding up.

## The probes, and what they printed

**The curated base.** The count in `contrastiveBase.params`, through the project's own loader
rather than a grep, because a grep for `- id:` counts 44 across the three files only by accident
of formatting:

```bash
uv run python -c "from pathlib import Path; from teacher_charles.teaching.contrastive import load_items; print(len(load_items(Path('data/contrastive'))))"
# 44
```

**The projection parses.** twowayflow's own tree-sitter parser over this file, with a control,
run from a throwaway test inside the twowayflow checkout and deleted afterwards:

```
nodes 30 edges 58
unknownLinks []
diagnostics []
kinds {"source":5,"transform":17,"join":1,"filter":1,"sink":6}
CONTROL (one edge retargeted at an undeclared node):
  [{"code":"dangling-edge","severity":"warning","message":"Edge to references undeclared node \"sttTypo\"."}]
```

Counted a second way, off the source rather than the parse, because two numbers that should
agree are worth more than either alone — `grep -c '^  \.edge('` gives 58 and
`grep -c '^  \.node('` gives 30.

The type system agrees independently: `tsc --build` over the examples workspace accepts the file,
and rejects the same mutation with `TS2345: Argument of type '"sttTypo"' is not assignable`. The
control is the point — a green check with nothing that can fail it establishes nothing.

**Which modules a door actually reaches.** An AST import graph, rooted at the entry points, run
to check that no node of the diagram sits in code nothing reaches:

```
entry points cli app    32 modules, 29 reachable
                         unreachable: teacher_charles.store
                                      teacher_charles.subtitles
                                      teacher_charles.teaching.vocabulary
```

`teacher_charles.store` is the package `__init__`, which only re-exports — `app.py` imports
`.store.db` and `.store.repo` directly. The other two are findings, and both are recorded below
rather than quietly fixed.

### Finding: an analysis outage reads as a drill hit

`CLAUDE.md` states the guard this way, and it is the one it calls the guard that matters most:

> an analysis that did not run graduates nobody — without it, an analyzer that is down would turn
> into a success, because it produces the same absence of corrections.

The guard in `TurnRunner._observe_schedule` is `if analysis is None: return ()`. But
`Analyzer.analyze` **never returns `None`**. On a timeout or a provider failure it returns
`Analysis(degraded=True)`; when the base raised no candidate it returns `Analysis(skipped=True)`.
Both are `Analysis` objects with no findings — which, inside a drill, is exactly the shape of a
hit. `None` happens only when there is no analyzer at all, or when the task raised or was
cancelled, and `test_sem_analise_nao_ha_revisao` covers the first of those.

Measured, driving a real drill turn whose analysis provider raises `ProviderUnavailable` while the
lesson streams normally. The utterance matters and the first attempt at this probe got it wrong
(see below): it has to be text the base **matches**, or the analyzer returns `skipped=True`
without ever calling the model, and the degraded path is never reached. So the student writes the
drill's own calque — the worst case available, because the rule being trained is the rule they
just missed:

```
student: "I have 35 years"    (base matches calque.age-have; the drill is training it)
analysis: degraded — "o modelo falhou na análise"
  review: hit=True source=drill box 0->1
  schedule: box=1 streak=1 reviews=1 lapses=0
```

The student missed the rule and the schedule credited them with getting it right. Three of those
graduate it, which is the failure the quoted sentence says cannot happen.

**Fixed, after this was measured and recorded.** The order is in the git history and worth
keeping: commit `cc4ebf3` recorded the finding while the application was untouched; the fix came
after, as its own commit, once the repository's owner decided a degraded analysis should be
silent. The guard is now `analysis is None or not analysis.is_evidence`, and
`Analysis.is_evidence` is where the reasoning sits.

`skipped=True` is deliberately **not** covered by it. There the base ran and matched nothing,
which is a poor observation of the text rather than no observation — and it is the
already-documented blind spot, pinned in
*"test_uma_resposta_vazia_de_conteudo_conta_como_acerto"* and recorded as a Phase 2 debt.
Narrowing `is_evidence` to exclude it as well turns that test red, which is how the scope is held.

The new guard was mutation-tested three ways rather than trusted: reverting the condition to
`analysis is None` turns *"test_a_degraded_analysis_graduates_nobody"* red, so does making
`is_evidence` return `True`, and widening it to exclude `skipped` turns the blind-spot test red
instead. A guard whose test cannot be made to fail is not evidence of anything.

### Finding: the door is a string, so an import graph reads this project as dead

Rooted only at the console script — `teacher_charles.cli:main`, the one entry point
`pyproject.toml` declares — the same probe says **3 of 32 modules are reachable** and calls
`app`, `teaching`, `store`, `llm` and `voice` unreachable. Nothing is wrong with the code. The
edge from `cli` to `app` is `uvicorn.run("teacher_charles.app:app", ...)` at
[cli.py:92](../../src/teacher_charles/cli.py#L92) — a string, invisible to every static import
graph.

So the probe's entry-point list, not its output, is the thing that has to be right. Naming `app`
as a door too gives 29 of 32. Anyone who later runs a dead-code sweep over this repository and
trusts the first number will delete the product.

### Finding: two modules are built, tested, and on no product path

`teacher_charles.subtitles` — the SubRip reader added for the corpus work — is imported by
nothing except its own test:

```bash
grep -rln "subtitles" --include='*.py' src tests scripts
# src/teacher_charles/subtitles.py
# tests/test_subtitles.py
```

`teacher_charles.teaching.vocabulary` is the same shape, and it is newer: 398 lines of module,
182 lines of curated `data/vocab/items.yaml`, a JSON schema, a measurement script and 555 lines
of tests, imported by `scripts/dump_vocab_schema.py` and its own test and by nothing the turn
runs.

A grep says otherwise, and that is worth keeping:

```bash
grep -rln "vocab" --include='*.py' src tests scripts
# ...src/teacher_charles/teaching/analyzer.py  <- "vocabulary" is an error-type STRING
# ...src/teacher_charles/teaching/level.py     <- the word, in a comment
```

Both hits are the word, not the module — `analyzer.py:60` lists `"vocabulary"` among
`ERROR_TYPES`. Resolving through imports rather than through the name is what separates them:

```bash
grep -rn "from .*vocabulary import" --include='*.py' src tests scripts
# scripts/dump_vocab_schema.py
# tests/test_vocabulary.py
```

So neither a `corpus/ → subtitles.py → data/` lane nor a vocabulary node is drawn. The 44
contrastive items were written by hand, not derived, and the vocabulary items reach no prompt:
drawing either would be drawing a pipeline the product does not run. Both are tools waiting to be
wired in, which is a true thing to say about them and a false thing to draw.

A third, smaller mismatch: `CLAUDE.md` says the fast suite is 721 tests. At `96e0249` it is 852
passed and 13 skipped.

## What this notation cannot carry

A dataflow graph's primitive is the **data dependency**. Chosen deliberately — the question this
diagram answers is "what does each step need in order to run", and that is the question a reader
of `teaching/turn.py` actually has. But three claims this project cares about are claims about
**order**, and no amount of care with the arrows will express them:

- **The student's chunks have already left the process before `analyze` has produced anything.**
  `stream → sseFrames` and `analyze → verify → sseFrames` are both drawn, and the graph is silent
  on the fact that the first frames are on the wire while the second path is still running. Here
  that is the design and not a defect — the correction annotates what the student said, it does
  not interrupt what the teacher is saying — but the graph cannot say so, and in a system where it
  *were* a defect it would look identical.
- **A review is two writes and no transaction**, `rule_review` before `rule_schedule`, chosen in
  that order because the state can be rebuilt from the event and not the other way round. The
  graph shows one edge into one sink.
- **The order of the system messages is the measured design** — persona, capabilities, level,
  mode, drill data, then the notes at the very end, inside the student's message. `buildMessages`
  is one fan-in node with five incoming edges, and fan-in has no order.

The notation for all three is a **sequence diagram with lifelines**, and it is the obvious next
thing to draw here. It is not drawn in this pass.

Two more claims about order came out of the review of this document, and they belong in the same
place because the same limit applies — neither is visible in the graph, and neither was
investigated here:

- **Two tabs posting to one session.** Nothing serialises `/api/chat` per session. Two requests
  can each append their student row before either model call finishes, so the later prompt sees
  both utterances and the replies land in completion order. The graph draws one turn; it has no
  way to draw two of them interleaved.
- **A level change is two writes.** `set_level` then `log_change`, and the second is what closes
  the window via `at_signal_id`. If the first lands and the second does not, the level has moved
  with nothing marking the signals as spent. `observeLevel → levelRows` is one edge.

Both are the shared-connection rule playing out — no request path opens a multi-statement
transaction — and both are ordering claims a dataflow graph cannot carry.

## What was deliberately not drawn

- **The corpus lane** (`corpus/ → subtitles.py → data/`) and **the vocabulary lane**
  (`data/vocab/items.yaml → teaching/vocabulary.py`): no entry point reaches either, see the
  finding above.
- **The `scripts/measure_*.py` family.** Eight measurement harnesses that drive the same pipeline
  from outside. They would double every node for no new dependency; what they produced is already
  in the `params` of the nodes they measured.
- **An ER diagram of the nine tables.** `store/` has no ORM and the schema is five numbered
  `.sql` files; the interesting content is the nullable `correction.rule_id` and
  `level_change.at_signal_id`, both of which are a sentence, not a picture.
- **A class diagram.** The product's units are modules and protocols (`ChatProvider`,
  `SpeechToText`, `TextToSpeech`) with one implementation each plus a fake; the inheritance tree
  is two levels deep everywhere and would draw as a list of records.
- **The `/progress` screen's read path.** It is query, aggregate, render — a report over the rows
  this diagram already shows being written.
- **Phase 2's phoneme evaluator.** E9 is an unpassed gate and the evaluator does not exist. An
  unbuilt subsystem is the easiest thing in a codebase to draw, because nothing contradicts you.

## Expiry

This document describes `96e0249`. The counts (30 nodes, 58 edges, 44 curated items, 852 tests)
are pinned to it, each with the probe that produced it. The sequence diagram named above as
missing is missing *as of that revision*; when someone draws it, add the row rather than editing
this section away — the argument for why the dataflow graph could not carry those three claims is
the part worth keeping.

### What the first version of this document got wrong

Kept rather than quietly corrected, because the causes recur:

| What it said | How it got there |
|---|---|
| “nothing graduates when the analysis did not run”, citing a test | the cited test is real and passes; it covers `analyzer=None`, and the sentence had generalised to “did not run”. The instrument behind the claim was the defect — the claim was never measured against a *degraded* analysis until a reviewer asked |
| the finding above, in `cc4ebf3`, labelled its evidence `DEGRADED` | the probe used *”I am 35 years old.”* — correct English, so the base matched nothing, so the analyzer returned `skipped=True` and never called the model. The `hit=True` it printed was the known `skipped` blind spot wearing the new finding's name. The conclusion survived re-measurement with base-matching text; the evidence for it did not, and a reader checking the probe would have found the wrong mechanism |
| `_Verifier` “keeps only findings citing a rule the curated base matched there” | read off the first branch of `_rule_for` and stopped; the second branch accepts an id the model claimed, if the base holds it |
| `maxFindings: { correction: 5, other: 2 }` | true as a map of mode to cap, and read by a reviewer as "five corrections plus two others". Ambiguity in a params key is a wrong number waiting to be quoted |
| `matchBase → verify` and `matchBase → analyze` | the analyzer recomputes `base.match` instead of consuming those results, so the graph drew a dependency the code does not have — §1's failure, an accurate drawing of a path that is never taken |
| eight incoming edges invented for `buildMessages` (first "six", then "nine") | both written from memory of the node list instead of counted; `grep -c` says five |
| `lever: "the idle Intel iGPU"` beside `tokPerS: "7-8"` | credits the iGPU for generation, which the measurement denies (0.91× on the 3B — a tie). The 7.4s → 0.9s came from prompt evaluation, 8–14× faster on the Vulkan backend |

Five of the six were caught by review, not by the probes in this document — the probes checked
that the graph parses, that its citations resolve and that its nodes are reachable, and none of
them could check whether a sentence means what the code does.

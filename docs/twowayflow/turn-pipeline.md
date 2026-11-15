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

To look at it:

```bash
code /home/mpb/Documents/GitHub/twowayflow   # F5, then open the .flow.ts and run "twowayflow: Open Flow Canvas"
```

Read without the canvas it is still a flat list: every node on one line, every edge on one line,
the prose for each box in its `description`.

## What it says, in one paragraph

29 nodes, 49 edges. Five sources (the prompt files, the curated base, the student's utterance, the
profile-and-session row, the microphone), sixteen transforms, one fan-in (`buildMessages`), one
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
| no `verify → observeSchedule` edge when the analysis did not run — nothing graduates | *"test_sem_analise_nao_ha_revisao"* |
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
nodes 29 edges 49
unknownLinks []
diagnostics []
kinds {"source":5,"transform":16,"join":1,"filter":1,"sink":6}
CONTROL (one edge retargeted at an undeclared node):
  [{"code":"dangling-edge","severity":"warning","message":"Edge to references undeclared node \"sttTypo\"."}]
```

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
  is one fan-in node with six incoming edges, and fan-in has no order.

The notation for all three is a **sequence diagram with lifelines**, and it is the obvious next
thing to draw here. It is not drawn in this pass.

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

This document describes `96e0249`. The counts (29 nodes, 49 edges, 44 curated items, 852 tests)
are pinned to it, each with the probe that produced it. The sequence diagram named above as
missing is missing *as of that revision*; when someone draws it, add the row rather than editing
this section away — the argument for why the dataflow graph could not carry those three claims is
the part worth keeping.

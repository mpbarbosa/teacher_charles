import { flow } from "@twowayflow/dsl";

/**
 * One turn of Teacher Charles, as data: `POST /api/chat` from the student's
 * utterance to the frames the page draws and the rows SQLite keeps.
 *
 * Projected from the code at `96e0249`, not from the plans in `docs/`. The node
 * ids are the names the Python uses (`matchBase`, `ruleHint`, `buildMessages`,
 * `verify`, `observeSchedule`), so a reader can go from a box to a symbol, and
 * `params` carry this project's own measured numbers rather than round ones.
 *
 * What the graph is for, and what it is not: a dataflow graph's primitive is a
 * **data dependency**, so it answers "what does this step need in order to
 * run". It has no lifelines, so it cannot say what has already left the process
 * — and in this pipeline that matters twice, because the student's chunks are on
 * the wire before `analyze` has produced anything, and because a review is two
 * unwrapped writes whose order is a deliberate choice. Both are recorded in
 * `turn-pipeline.md` §"What this notation cannot carry", with the sequence
 * diagram they would need instead.
 *
 * Three things here that are easy to read backwards:
 *
 *  1. **The cycle is the product.** `observeSchedule -> scheduleRows -> queue ->
 *     sessionRow -> drillTarget -> buildMessages` is E13: what the student got
 *     wrong today changes the prompt of a *later* session, never this one. The
 *     hop through `sessionRow` is not decoration — the rule is chosen once, at
 *     `POST /api/mode`, and stored in `session.drill_rule_id`.
 *  2. **The microphone does not feed the turn.** `stt -> studentText` fills the
 *     text field; the student still presses send. An STT slip that became a turn
 *     would have Charles correcting a mistake the student never made.
 *  3. **An absent edge carries a claim, and this one is weaker than it looks.**
 *     `verify -> observeSchedule` is labelled for the silence as well as the
 *     error: a hit in the drill *is* the absence of a finding citing the rule.
 *     The guard meant to stop a model outage from reading as learning tests
 *     `analysis is None` — and a degraded analysis is not None, so an outage
 *     during a clean drill turn *does* record a hit. Measured while drawing
 *     this; see `turn-pipeline.md`.
 *
 * Upstream: https://github.com/mpbarbosa/teacher_charles — `teaching/turn.py`
 * runs this, `app.py::chat` turns the events into SSE frames, and `CLAUDE.md`
 * holds the invariants each `description` below is a short form of.
 */
export default flow("teacher-charles-turn")
  // --- what is on disk before the student says anything ------------------- //
  .node("promptFiles", {
    kind: "source",
    at: [0, 0],
    label: "prompts/*.md",
    description:
      "The persona and the partials, loaded by prompts.py with an mtime-keyed cache: editing the persona takes effect on the next turn without a restart.",
    params: { loader: "prompts.py", cache: "mtime", partials: ["app", "level", "mode"] },
  })
  // Lemma and pattern matching over curated YAML, not vector search: ~44 items
  // fit in memory, the matching is deterministic and debuggable, and each item
  // gets a test. Embeddings are a Phase 2 note.
  .node("contrastiveBase", {
    kind: "source",
    at: [0, 200],
    label: "data/contrastive/*.yaml",
    description:
      "The curated contrastive base: the only source of grammar in the product. A rule id the base does not hold cannot reach the student.",
    params: { files: 3, items: 44, calque: 17, falseFriend: 17, grammar: 10, at: "96e0249" },
  })
  .node("studentText", {
    kind: "source",
    at: [0, 520],
    label: "the student's utterance",
    description:
      "The message in the JSON body of POST /api/chat. POST and not EventSource, because EventSource is GET-only and what the student said does not belong in a query string.",
    params: { route: "POST /api/chat", body: "json", maxChars: 4000 },
  })
  .node("sessionRow", {
    kind: "source",
    at: [0, 860],
    label: "student + session",
    description:
      "The active profile and its open session: CEFR level, mode, and the one rule the drill is training. The active profile is the highest student.seen_seq, a counter and not a timestamp.",
    params: { tables: ["student", "session"], active: "max(seen_seq)", modes: 3 },
  })
  .node("micAudio", {
    kind: "source",
    at: [0, 1400],
    label: "push-to-talk bytes",
    description:
      "The recording posted raw to /api/stt — one blob, no multipart. The engine decodes the container (webm/opus, ogg, wav).",
    params: { route: "POST /api/stt", body: "raw bytes", optional: "voice extra" },
  })
  // --- the two ends of Phase 1.5 ----------------------------------------- //
  // Accuracy with a Brazilian accent is this phase's open debt: the audio in
  // the tables is synthetic, so the 0% WER is a floor and not an expectation.
  .node("stt", {
    kind: "transform",
    at: [340, 1400],
    label: "WhisperSTT",
    description:
      "faster-whisper on the CPU, returning text and nothing else. It records no turn at all, which is the safeguard that does not depend on which STT model is loaded.",
    params: { model: "base.en", medianS: 0.91, av: "<19 (measured incompatibility)" },
  })
  // --- the prompt of the reply ------------------------------------------- //
  // There are TWO match passes per turn, and this is only the first. The
  // analyzer does not consume this one — it calls `self._base.match(text)`
  // again for its own candidates and for the verifier. Drawn as one node, the
  // graph would claim a dependency the code does not have; drawn as two, it
  // shows the same regex pass running twice. At microseconds that is a cost
  // nobody needs to pay down, but it is what the code does.
  .node("matchBase", {
    kind: "transform",
    at: [340, 200],
    label: "base.match (for the hint)",
    description:
      "Compiled regexes over the normalised utterance, in microseconds, run by TurnRunner.rule_hint. Its only consumer is the curated hint that rides with the student's message.",
    params: { cost: "microseconds", returns: "spans + rule_ids", consumedBy: "rule_hint only" },
  })
  .node("analyzeMatch", {
    kind: "transform",
    at: [1020, 1220],
    label: "base.match (inside the analyzer)",
    description:
      "The analyzer's own pass over the same text, recomputed rather than passed in. It produces the candidates block for the prompt, the spans the verifier checks against, and the decision not to call the model at all when nothing matched.",
    params: { sameInputAs: "matchBase", recomputed: true, maxCandidates: 4 },
  })
  .node("appendStudent", {
    kind: "transform",
    at: [340, 520],
    label: "turn.append(student)",
    description:
      "Writes the student's row before the model is called, so a generation that fails does not lose what the student said. Returns the turn_id every later row hangs off.",
    params: { table: "turn", before: "the model call", returns: "student_turn_id" },
  })
  .node("drillTarget", {
    kind: "transform",
    at: [340, 860],
    label: "DrillPlanner.target_for",
    description:
      "Resolves the session's rule into a target: the curated item, the student's own past sentences for it, and where it sits on the review ladder.",
    params: { studentExamples: 3, picksRule: false, source: "session.drill_rule_id" },
  })
  .node("ruleHint", {
    kind: "transform",
    at: [680, 200],
    label: "rule_hint",
    description:
      "The curated explanation handed to the teacher so the prose comes from the reviewed rule instead of a 4B model's improvisation. Deterministic and free — no model is asked.",
    params: { maxRules: 3, deterministic: true },
  })
  .node("history", {
    kind: "transform",
    at: [680, 520],
    label: "TurnRepo.history",
    description:
      "The last turns of this session, already including the row just written — which is why build_messages does not append the new message a second time.",
    params: { turns: 20, reason: "prompt evaluation costs real time, not a context limit" },
  })
  // Text in, text out, no audio anywhere: that is why the G2P lives in
  // teaching/ and its wheels are base dependencies, not part of the voice
  // extra. Voice is off by default and the line must not vanish with it.
  .node("spellStudent", {
    kind: "transform",
    at: [680, 700],
    label: "G2P (student)",
    description:
      "espeak-ng through phonemizer — the same G2P Kokoro speaks with, so the symbol on screen is the symbol the loudspeaker pronounces. Emitted before the model is called.",
    params: { engine: "espeak-ng", perCallMs: 0.3, scope: "one line per bubble" },
  })
  .node("buildMessages", {
    kind: "join",
    at: [1020, 340],
    label: "build_messages",
    description:
      "Persona, app capabilities, level, mode, drill data, history, and then the curated hint and the drill request inside the student's last message. The beginning is ordered by stability, the end by effectiveness.",
    params: {
      prefixOrder: ["persona", "app", "level", "mode"],
      notesGo: "inside the last user message",
      noSystemMessageAfterHistory: true,
    },
  })
  // Streaming is not an ornament: it is what makes 7-8 tok/s tolerable. Do not
  // accumulate the response before relaying it.
  .node("stream", {
    kind: "transform",
    at: [1360, 520],
    label: "provider.stream",
    description:
      "The teacher's reply, token by token, from Ollama behind llm/base.ChatProvider. Nothing above this node knows whether the model is local or in the cloud.",
    params: {
      model: "qwen3:4b",
      firstWordS: 0.9,
      wasS: 7.4,
      // What bought the 7.4s -> 0.9s is PROMPT EVALUATION on the Arc iGPU
      // through Ollama 0.35's Vulkan backend: 8-14x on a 3B, 13-18x on a 7B.
      // Generation is a different story and the iGPU does not carry it —
      // measured 0.91x on the 3B (a tie) and 1.50x on the 7B. So the tok/s
      // below is the CPU-class figure the whole design is budgeted against,
      // not something the iGPU improved.
      lever: "prompt eval on the iGPU (Ollama 0.35 Vulkan)",
      promptEvalSpeedup: "8-14x (3B)",
      generationSpeedup: "0.91x (3B) — a tie",
      tokPerS: "7-8",
    },
  })
  .node("appendTeacher", {
    kind: "transform",
    at: [1700, 520],
    label: "turn.append(teacher)",
    description: "The joined reply, written once the stream closes.",
    params: { table: "turn", returns: "teacher_turn_id" },
  })
  .node("spellTeacher", {
    kind: "transform",
    at: [1700, 700],
    label: "G2P (teacher)",
    description:
      "Runs once the stream has closed, on the joined reply, and the whole reply at once rather than sentence by sentence as the TTS does. At ~0.3 ms there is nothing to bring forward by segmenting, and a per-sentence line would be patched where the last word of one sentence and the first of the next never meet.",
    params: { scope: "the whole reply", after: "the last chunk", perCallMs: 0.3 },
  })
  // --- the analysis, and the guard that keeps it honest ------------------ //
  // Conversation first, analysis after the last chunk: Ollama serves one
  // request at a time on this machine, so "parallel" was fiction and only
  // delayed the first word (11.2s against 0.7s).
  .node("analyze", {
    kind: "transform",
    at: [1360, 1040],
    label: "Analyzer.analyze",
    description:
      "The second model call, at temperature 0, asked for JSON only. The one retry is for a reply that was not valid JSON or not in contract — a timeout or a provider failure breaks out at once, with no second attempt. It never raises: a dead analysis degrades to no correction, because the student's turn is worth more than its correction.",
    params: {
      temperature: 0.0,
      maxTokens: 320,
      timeoutS: 45.0,
      retriesOn: "unparsable or out-of-contract JSON only",
      attempts: 2,
      medianS: 8.4,
      candidates: 4,
    },
  })
  // E4's guard against the small model inventing grammar, and it has not
  // changed. What changed is that the barred finding is now kept.
  .node("verify", {
    kind: "filter",
    at: [1700, 1040],
    label: "_Verifier",
    description:
      "Locates each claimed span in the student's own text, then decides the rule itself: the rule the base matched over that span, or — failing that — the id the model claimed, accepted only because it exists in the base. An id that exists nowhere is discarded, and with require_curated_rule on a finding with no rule at all is refused.",
    params: {
      requireCuratedRule: true,
      // One cap per turn, whose value is the mode's — not 5 plus 2.
      maxFindingsByMode: { conversation: 2, correction: 5, drill: 2 },
      // The second branch of `_rule_for`, which is deliberate upstream and
      // worth naming here: a rule the base did not match at that span can
      // still be cited, if the model named one the base holds.
      acceptsClaimedIdPresentInBase: true,
    },
  })
  // --- what the turn leaves behind --------------------------------------- //
  .node("corrections", {
    kind: "sink",
    at: [2040, 960],
    label: "correction",
    description:
      "The structured correction: span, corrected form, error type, explanation, and a nullable rule_id where NULL marks 'came from the model alone'. This table is the error log the progress screen reads.",
    params: { table: "correction", ruleIdNullable: true, since: "the first migration" },
  })
  .node("gaps", {
    kind: "sink",
    at: [2040, 1200],
    label: "unverified_finding",
    description:
      "The curation backlog: each row is an error in a real student's text that the curated base does not cover. Written after the corrections, because losing a curation row costs a note the next similar turn offers again.",
    params: { table: "unverified_finding", readBy: "scripts/curation_queue.py", neverRaises: true },
  })
  .node("observeLevel", {
    kind: "transform",
    at: [2040, 700],
    label: "LevelTracker.observe",
    description:
      "A deterministic rubric — sentence length, long words, connectives, structures, and errors that cite a curated rule. The level is decided by code, never by the model, and the new one applies from the next turn.",
    params: { agreeingTurns: 4, stepsAtATime: 1, estimatesUpTo: "C1", countsOnly: "curated errors" },
  })
  .node("levelRows", {
    kind: "sink",
    at: [2380, 700],
    label: "level_signal + level_change",
    description:
      "One signal row per turn and one change row per move. level_change.at_signal_id closes the window, so the same four turns cannot move the level again next turn.",
    params: { tables: ["level_signal", "level_change"], window: "closed by at_signal_id" },
  })
  .node("observeSchedule", {
    kind: "transform",
    at: [2380, 1120],
    label: "_observe_schedule",
    description:
      "Moves the review schedule with what this turn showed. An error by the rule counts in any mode; a clean turn counts as a hit only inside the drill, over the rule being trained, because there the exercise built the opportunity.",
    params: {
      ladderDays: [0.5, 1, 3, 7, 16, 35],
      graduatedStreak: 3,
      // The guard is `analysis is None`, which is NOT the same as "the analysis
      // ran". A degraded analysis (timeout, provider down) is an Analysis
      // object with no findings, so inside a drill it reads as a hit —
      // measured, see turn-pipeline.md §"Finding: an analysis outage reads as
      // a drill hit". `None` happens only with no analyzer at all, or a task
      // that raised or was cancelled.
      noReviewWhen: "analysis is None",
      notTheSameAs: "the analysis ran",
    },
  })
  .node("scheduleRows", {
    kind: "sink",
    at: [2720, 1120],
    label: "rule_review + rule_schedule",
    description:
      "The event, then the state. Two writes and no transaction, by the shared-connection rule, and that order is chosen because the state can be rebuilt from the event and not the other way round.",
    params: { tables: ["rule_review", "rule_schedule"], order: "event before state" },
  })
  // The swap E13 made: frequency only goes up, so the most-missed rule in the
  // history stayed the most-missed one after the student had got it right ten
  // times, and the drill never let it go.
  .node("queue", {
    kind: "transform",
    at: [2720, 880],
    label: "Scheduler.queue",
    description:
      "The drill queue as a schedule, not a ranking: overdue first, recently-drilled demoted, oldest due date next, and the error count left as the last tiebreak between equally urgent rules.",
    params: {
      sortKey: ["not due", "recently drilled", "due_at", "-times", "rule_id"],
      neverEmptyWithCandidates: true,
    },
  })
  // --- what the student gets --------------------------------------------- //
  .node("segment", {
    kind: "transform",
    at: [1700, 1400],
    label: "SentenceAssembler",
    description:
      "Cuts the stream into sentences so synthesis of sentence 1 runs while the model writes sentence 2. A boundary also counts at the end of a chunk, which buys back ~130 ms per sentence at the only point where the student waits in silence.",
    params: { minChars: 12, closesAtChunkEnd: true, waitsOn: "a digit before the period" },
  })
  .node("tts", {
    kind: "transform",
    at: [2040, 1400],
    label: "KokoroTTS",
    description:
      "One request per sentence to /api/tts, serialised by the engine's own lock so concurrent requests queue instead of fighting for CPU. The audio never travels on the SSE.",
    params: { route: "POST /api/tts", voice: "bm_george", firstSpokenWordS: [5.44, 7.35] },
  })
  .node("audioQueue", {
    kind: "sink",
    at: [2380, 1400],
    label: "voice.js playback",
    description:
      "Plays by index, not by arrival: the requests to /api/tts resolve out of order, and arrival order would scramble the teacher's speech the first time the network hesitated.",
    params: { playsBy: "index", notBy: "arrival" },
  })
  .node("sseFrames", {
    kind: "sink",
    at: [2380, 520],
    label: "SSE frames",
    description:
      "data: lines over a POST response — chunk, phonetics, analysis, level, sentence, done, error. X-Accel-Buffering: no is there so a proxy on the path cannot hold the body and cancel out the streaming.",
    params: {
      types: ["chunk", "phonetics", "analysis", "level", "sentence", "done", "error"],
      transport: "SSE over POST",
      header: "X-Accel-Buffering: no",
    },
  })
  // --- edges -------------------------------------------------------------- //
  // The transcript fills the text field; the student still presses send. STT
  // makes mistakes, and an STT error that became a turn would have Charles
  // correcting an error the student did not make.
  .edge("micAudio", "stt", { label: "bytes" })
  .edge("stt", "studentText", { label: "into the text field, never a turn" })
  .edge("studentText", "appendStudent")
  .edge("studentText", "matchBase", { label: "normalised text" })
  .edge("studentText", "spellStudent")
  .edge("studentText", "analyze", { label: "the utterance, verbatim" })
  .edge("studentText", "analyzeMatch", { label: "normalised text" })
  .edge("contrastiveBase", "matchBase", { label: "compiled patterns" })
  .edge("contrastiveBase", "analyzeMatch", { label: "compiled patterns" })
  .edge("contrastiveBase", "verify", { label: "which ids exist at all" })
  .edge("contrastiveBase", "drillTarget", { label: "curated item" })
  .edge("matchBase", "ruleHint", { label: "matches" })
  .edge("analyzeMatch", "verify", { label: "spans that matched" })
  // With require_curated_rule on, no candidate means no model call: nothing the
  // model returned could reach the student, so it is not asked. Which is also
  // why every curation row comes from a turn where some other rule matched.
  .edge("analyzeMatch", "analyze", { label: "candidates; none, no call" })
  .edge("promptFiles", "buildMessages", { label: "persona + partials" })
  .edge("promptFiles", "analyze", { label: "analysis prompt" })
  .edge("sessionRow", "buildMessages", { label: "cefr_level, mode" })
  .edge("sessionRow", "drillTarget", { label: "session.drill_rule_id" })
  .edge("sessionRow", "observeLevel", { label: "student_id" })
  .edge("sessionRow", "observeSchedule", { label: "student_id, mode" })
  // The analysis is scoped by the profile too: the level goes in its prompt,
  // and the mode sets the cap on how many findings may come back.
  .edge("sessionRow", "analyze", { label: "cefr_level, and the mode's cap" })
  // Every turn row is owned by a session, and the prompt's history is read
  // back by that same id. Without these the writes look session-less.
  .edge("sessionRow", "appendStudent", { label: "session_id" })
  .edge("sessionRow", "appendTeacher", { label: "session_id" })
  .edge("sessionRow", "history", { label: "session_id" })
  .edge("appendStudent", "history", { label: "the row just written" })
  .edge("appendStudent", "corrections", { label: "turn_id" })
  .edge("appendStudent", "gaps", { label: "turn_id" })
  .edge("appendStudent", "observeLevel", { label: "turn_id" })
  .edge("appendStudent", "observeSchedule", { label: "turn_id" })
  .edge("history", "buildMessages", { label: "last 20 turns" })
  .edge("ruleHint", "buildMessages", { label: "inside the last user message" })
  .edge("drillTarget", "buildMessages", { label: "drill block + reminder" })
  .edge("buildMessages", "stream", { label: "messages" })
  .edge("stream", "sseFrames", { label: "chunk, relayed at once" })
  .edge("stream", "appendTeacher", { label: "the joined reply" })
  .edge("stream", "segment", { label: "voice on only" })
  .edge("appendTeacher", "spellTeacher", { label: "reply" })
  .edge("spellStudent", "sseFrames", { label: "IPA, student bubble" })
  .edge("spellTeacher", "sseFrames", { label: "IPA, teacher bubble" })
  .edge("segment", "sseFrames", { label: "sentence + index" })
  .edge("segment", "tts", { label: "one request per sentence" })
  .edge("tts", "audioQueue", { label: "WAV, played by index" })
  .edge("analyze", "verify", { label: "the model's JSON" })
  .edge("verify", "corrections", { label: "cites a curated rule" })
  .edge("verify", "gaps", { label: "cites none" })
  // Analysis.as_dict does not expose `rejected`: material for the curator is
  // not material for the student.
  .edge("verify", "sseFrames", { label: "findings only, never the rejected" })
  .edge("verify", "observeLevel", { label: "errors citing a curated rule" })
  // The label carries the asymmetry: an error by the rule in any mode, and —
  // inside the drill, over the rule in training — the absence of one.
  .edge("verify", "observeSchedule", { label: "which rules erred, and the silence that is a hit" })
  .edge("observeLevel", "levelRows", { label: "signal, and a change if it moved" })
  .edge("observeLevel", "sseFrames", { label: "level, every turn" })
  .edge("observeSchedule", "scheduleRows", { label: "event, then state" })
  .edge("scheduleRows", "queue", { label: "due_at, box, streak" })
  .edge("corrections", "queue", { label: "the last tiebreak" })
  // The rotation term: queue() asks which rules the last few drill sessions
  // already trained, so one picked and left unanswered is not picked again.
  .edge("sessionRow", "queue", { label: "the last drilled rules" })
  // Resolving the session's rule into a target reads the student's own past
  // sentences for it, and where it sits on the ladder.
  .edge("corrections", "drillTarget", { label: "3 of the student's own spans" })
  .edge("scheduleRows", "drillTarget", { label: "box, streak, due_at" })
  // The rubric scores the utterance itself; the analysis only supplies the
  // one negative signal. Without this edge the level looks derivable from
  // the corrections alone.
  .edge("studentText", "observeLevel", { label: "the text the rubric scores" })
  // The back edge, and the point of E13: the rule is chosen once per session,
  // at POST /api/mode, and lands in session.drill_rule_id. So what the student
  // got wrong today shapes a LATER session's prompt, never this turn's.
  .edge("queue", "sessionRow", { label: "next session's rule (POST /api/mode)" });

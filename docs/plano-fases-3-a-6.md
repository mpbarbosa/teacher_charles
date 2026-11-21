# Teacher Charles — after Phase 2: phases 3 to 6

> **State: order accepted (2026-11-21), nothing started.** This document turns the four lines of
> `docs/plano-fase-2.md` §8 into phases with deliverables, each with a
> definition of done. Every number quoted here comes from a measurement already
> on disk; every number this document *wants* is marked as a target, to be
> replaced by a measurement before it is relied on.

The rule that organized Phases 1.5 and 2 organizes these too: **a phase opens
with a measurement, not with integration.** Each one below states the question
that could cancel it, and the first deliverable is the one that answers it.

Deliverables continue the numbering: Phase 2 ends at E14, so these start at
**E15**.

---

## 0. The order, and why it is not §8's order

§8 lists turn-taking, prosody, the analysis debt and mobile/multi-user, in that
order. The order proposed here is different:

| Phase | Theme | Why here |
|---|---|---|
| **3** | The analysis debt | Touches every text turn with an error, needs no new weight or hardware, and its baseline is already measured (`docs/plano-fase-1.md` §5.5). The cheapest phase with the widest effect. |
| **4** | Conversation without a button | Its gate is the same STT accent measurement E14 waits for, so whichever runs first pays it for both. |
| **5** | Prosody and rhythm | Builds on E10's corpus of human takes and E11's recognizer; it needs those to mature first. |
| **6** | Mobile, multi-user, login, installer | Breaks three invariants that every earlier phase relies on (§6.1). It goes last so it breaks them once, over a finished product. |

**Accepted on 2026-11-21.** The reorder is a choice, not a dependency: phases 3 and 4 are
independent of each other and could run in parallel. Phase 5 depends on
Phase 2's E10 having more than three rules. Phase 6 depends on nothing
technically, but doing it earlier would make every later phase design for
multi-user.

---

## Phase 3 — The analysis debt

### 3.1 What is known

- The analysis costs a **median 10.1 s** per turn with an error and runs
  **after** the reply (`TC_ANALYSIS_AFTER_REPLY=true`), so it does not delay the
  first word; it delays the correction card (plano 1 §5.5).
- It is **generation-bound**: 63–83 tokens of JSON per call. The iGPU, which
  bought 8× on the first word, buys 1.3× here (plano 1.5 §7.7).
- A separate, bigger model was measured and **rejected**: a 7B for the analysis
  was 2.5× slower and fixed nothing, because the degradation it was meant to fix
  was the curated hint's position, not the analysis (plano 1 §5.5, "Dívida 2").
  `Analyzer` still accepts its own provider; nothing needs building to try one.
- The ceiling is `MAX_ANALYSIS_TOKENS = 320` and the deadline
  `DEFAULT_TIMEOUT_S = 45` (`teaching/analyzer.py`). Accuracy on the 31-sentence
  set: 27 of 31 on the default 3B, no false positive (plano 1 §5.5).

So "a better model just for it", as §8 phrases it, cannot mean a bigger model.
It has to mean **less generation**: fewer tokens per call, or fewer calls.

### 3.2 The question that opens the phase

> **How much of the 63–83 tokens is information the application already has?**

With `require_curated_rule` on, every correction that reaches the student cites
a `rule_id` from the base, and the base already holds the rule's wording and
explanation. If the model is also writing the explanation, those tokens are
paid for prose the student never sees, or sees in place of the curated text.

### 3.3 Deliverables

| | Deliverable | Done when |
|---|---|---|
| **E15** | **Opening measurement** — extend `scripts/measure_analysis.py` to split each reply's tokens by JSON field, and to time a second turn sent while the first turn's analysis is still running | The token share per field is recorded here. And one question is answered with a number: **does a quick next turn queue behind the analysis?** Ollama serves one request at a time on this model (plano 1 §5.2), so "the analysis runs after the reply" may only move the wait to the student's next first word. If it does, that is the real cost of the debt, not the correction card |
| **E16** | **The model chooses, the base writes** — the analysis returns only `rule_id`, span and corrected form; the explanation comes from the curated item. Structured output (Ollama's `format`, fed `_RawAnalysis.model_json_schema()` from `teaching/analyzer.py`) holds the smaller shape on the wire; malformed JSON was already rare (0 of 49 turns in E4), so this is about the shape, not the retry | Median analysis time measured by `scripts/measure_analysis.py` falls against E15's baseline, **accuracy holds at 27 of 31 or better, and false positives stay at zero** on the 18 correct sentences of plano 1 §5.2. A target of ≤ 3 s is a guess from the token count, not a commitment |
| **E17** | **No model when the base is sure** — when exactly one candidate's pattern matches and the item is marked deterministic, the correction is built without a model call | Each item that claims to be deterministic has a test showing it raises **no** candidate on a correct paraphrase. The share of corrections that took this path is measured on the curation queue's real turns. Zero false positives is the gate: a deterministic path that is wrong teaches the error with no model to blame |

### 3.4 The E13 debt belongs here

`docs/plano-fase-2.md` §6 records that inside the drill "did not err" and "did
not try" are the same observation: a student who answers *"yes"* three times
graduates the rule. That is an analysis problem, because the analyzer returns
errors and never "used the structure". Once E16 makes the analysis cheap, the
same pass can say whether the drill's target structure **appears** in the
reply — a pattern question the base can answer for many items without a model.

| | Deliverable | Done when |
|---|---|---|
| **E18** | **An attempt is evidence, a non-attempt is nothing** — a drill turn that never produced the target structure reviews nothing, the same way a degraded analysis reviews nothing | `test_uma_resposta_vazia_de_conteudo_conta_como_acerto` is inverted on purpose and renamed for what it now proves. `rule_review` history measures how many past hits were non-attempts (the data is already there through `turn_id`), and that number is recorded here before and after |

### 3.5 What this phase does not do

- No fine-tuning (decision 1.3 still holds).
- A cloud provider for the analysis stays an option the student chooses
  (`openai_compat` exists), never a default: the student's sentences would leave
  the machine.

---

## Phase 4 — Conversation without a button

### 4.1 The conflict this phase has to settle first

The push-to-talk button is not only an interface. It carries an invariant:
**the microphone transcript goes to the text field, not to the turn** (plano
1.5 §3.2, `test_nao_grava_turno_nenhum`). The student reads what Whisper heard
and fixes it before Charles sees it. That is the one safeguard that does not
depend on STT accuracy, and it exists because that accuracy with a Brazilian
accent has never been measured.

Hands-free conversation removes the moment where the student reads. So the
question is not "how to detect the end of speech". It is:

> **Is Whisper accurate enough on this student's voice for its transcript to
> become a turn without being read?**

If it is not, hands-free mode makes Charles correct errors the student never
made, which is the failure E3 and E4 exist to prevent. This is the same
measurement E14 waits for, and that is why the two share a gate.

### 4.2 What is already in place

- Silero VAD is already in the process, as `faster-whisper`'s input filter
  (`voice/whisper.py`), trimming silence from push-to-talk recordings. The
  docstring already names hands-free turn-taking as its natural next use.
- The voice turn costs **3.99 s** to the first spoken word with the iGPU, and
  the bottleneck is now the audio ends: STT 1.22 s plus TTS 1.29 s (plano 1.5
  §7.5). **Endpoint detection adds to this**: every millisecond of trailing
  silence the VAD waits for comes before the STT even starts.

### 4.3 Deliverables

| | Deliverable | Done when |
|---|---|---|
| **E19** | **Opening measurement** — Whisper WER on human Brazilian speech (the E14 gate, if E14 has not paid it), and the **distribution of pauses inside utterances** on the same recordings | WER is recorded per speaker, not as a single pooled number. Pause lengths inside sentences are recorded, because a B1 speaker who stops to search for a word must not be cut off: that distribution, not a constant, sets the end-of-turn silence. The decision to continue, change the design or stop is written with both numbers in hand |
| **E20** | **Endpointing** — a state machine (idle → listening → speech → trailing silence → submitted) in `voice/`, importing nothing from `teaching/` | `teaching/turn.py` does not change a line. The transcript policy E19 decided is enforced in code (for example: hands-free sends the transcript only when its confidence passes a measured floor, and otherwise drops it into the text field as push-to-talk does). The time from end of speech to first spoken word is measured with `scripts/measure_voice.py` and recorded next to §7.5's 3.99 s |
| **E21** | **Barge-in** — the student speaking stops Charles's playback queue in `voice.js` | Charles's own voice from the loudspeaker is never taken as the student's speech. Whether the browser's echo cancellation is enough with loudspeakers is measured, not assumed. If it is not, headphones are the documented requirement for hands-free mode and push-to-talk stays the default |

### 4.4 What this phase does not do

Push-to-talk stays, and stays the default until E19 says otherwise. Hands-free
is a mode the student turns on, because the measurement may hold for one voice
and not another (Phase 2 §6: the accent risk "returns whole for every new
speaker").

---

## Phase 5 — Prosody and rhythm

### 5.1 What the phoneme alignment cannot see

E11 judges **which** phoneme was said. It does not see **how long**, **how
loud** or **at what pitch**, and that is where several of a Brazilian speaker's
most audible English errors live:

- **Lexical stress**: *deVELop* said as *DEvelop*, the noun/verb pair *REcord*
  / *reCORD* (which `stress_shift.py` already handles on the G2P side).
- **Weak forms**: *to*, *for*, *of*, *can* said in their full citation form.
  The E12 practice session already met this in reverse: `/ə/` → `/uː/` in *to*
  was charged as an error by the expected text and is actually correct careful
  English (Phase 2 §5). Prosody rules will need the same `equivalences` field.
- **Rhythm**: Brazilian Portuguese is often described as tending to syllable
  timing and English as stress-timed, so every syllable gets similar length.

### 5.2 The question that opens the phase

> **On correct speech, does the measured stress agree with espeak's stress
> marks often enough that a disagreement means something?**

It is E8's floor and signal question, one layer up. The reference is free:
the IPA line already carries espeak's `ˈ` and `ˌ`. What is not free is
segment timing. CTC emissions are spiky, and whether the recognizer's alignment
gives syllable durations precise enough to compare is exactly what the opening
measurement has to find out.

Two honest expectations to test, not assume:

- **Lexical stress and weak forms are per-take questions**, about one word in
  one phrase, and may be judgeable.
- **Rhythm is probably not.** Rhythm metrics such as nPVI or %V are statistics
  over many syllables; a five-word phrase may not have enough to say anything.
  Rhythm may end up as a trend over many sessions on `/progress`, never as a
  per-take verdict.

### 5.3 Deliverables

| | Deliverable | Done when |
|---|---|---|
| **E22** | **Opening measurement** — duration, intensity and F0 per syllable on the E9/E10 human takes and on Kokoro, compared against espeak's stress marks | Floor (correct speech) and signal (deliberate misplaced stress) are recorded here, as in Phase 2 §2. **The one new dependency, a pitch tracker, is chosen by this measurement** and stays behind an extra, off by default, like `voice` and `syntax` |
| **E23** | **The prosody base** — curated items for stress and weak forms, in their own file next to `pronunciation.yaml` | Same admission rule as E10: no item goes in without a deliberate take that names it and a clean take that does not. Like `pron.*`, a prosody item **never enters the text base** and gets the same three tests `tests/test_pronunciation.py` holds |
| **E24** | **In the read-aloud drill** — prosody rules become drill targets on `/read-aloud`, with the same three takes and one review per session | The card speaks about the session ("the stress fell on the first syllable in 2 of 3 takes"), never as a score. `teaching/turn.py` does not change a line |

---

## Phase 6 — Mobile, multi-user, login, installer

### 6.1 The invariants this phase breaks

Everything so far assumes **one student, one browser, on the same machine as
the server**. Three invariants in `CLAUDE.md` rest on that:

1. **The active profile is the highest `student.seen_seq`**, which is global.
   Two browsers open at once would switch each other's student on every
   request.
2. **The server binds to `127.0.0.1`.** That is what makes having no login
   safe: only this machine can reach the student's data.
3. **The microphone works on `http://localhost`** because browsers treat
   localhost as a secure context. A phone reaching the PC at
   `http://192.168.x.x` is **not** a secure context, and `getUserMedia` is
   refused there: no voice, no read-aloud, no Follow me. Mobile on the local
   network needs HTTPS.

And one hardware fact: the machine generates ~9.3 tok/s on the 3B (plano 1
§1.3), and Ollama serves one request at a time. Two students talking at once
queue behind each other. **Multi-user here means several students taking turns
on one machine, not several at the same time** — unless a measurement shows the
queue is tolerable.

### 6.2 Deliverables

| | Deliverable | Done when |
|---|---|---|
| **E25** | **The student belongs to the browser, not to the database** — the active profile is bound per browser (a cookie naming the student), and `seen_seq` becomes only the default for a browser that has never chosen | The switch-and-switch-back profile test that broke the timestamp versions passes with **two clients at once**, each keeping its own student. The shared-connection rule still holds: no request opens a multi-statement transaction |
| **E26** | **Login, and a guard that makes it mandatory off loopback** — local accounts, passwords hashed with the standard library's `hashlib.scrypt`, sessions by cookie | **The server refuses to start on any host other than loopback with authentication off**, and a test holds that refusal in place (mutation-tested: delete the guard, the test goes red). On loopback, login stays optional, so the single-user install works exactly as today |
| **E27** | **The phone as a client** — HTTPS on the local network (a certificate the student installs once), a page that works at phone width, push-to-talk by touch (`voice.js` already listens to `pointerdown`) | The microphone works from a phone on the same network. The model still runs on the PC: the phone carries no weight. Time to first spoken word measured from the phone, next to plano 1.5 §7.5 |
| **E28** | **The installer** — one documented path from a fresh machine to a working teacher, including Ollama, the default model and the optional extras | Verified the way plano 1 §5.5 verified the README: a clean copy, no `.venv`, no caches, and the installer's steps in order. `scripts/check_env.py` is the final check of the install, not a separate step |

### 6.3 What this phase does not do

- **No cloud deployment.** The student's data lives in `XDG_DATA_HOME` on their
  own machine, and the product's premise is local inference. A hosted Charles
  would be a different product with a different privacy model, and needs its
  own decision.
- **No model on the phone.** The hardware budget that chose the 3B was measured
  on the PC; a phone is a different budget nobody has measured.

---

## 7. Decisions that are yours

1. ~~**The order** in §0.~~ **Decided on 2026-11-21: §0's order, the
   analysis debt first.** The alternative was §8's original order
   (turn-taking first), which would have left the analysis debt costing
   ~10 s per error turn for longer.
2. **What hands-free does when Whisper is unsure** (E20): drop the transcript
   into the text field, or ask the student by voice to repeat. The first keeps
   the invariant intact; the second keeps the hands free.
3. **Whether Phase 6 is wanted at all.** If the product stays one student on
   one machine, E25 alone (two profiles on one PC without stepping on each
   other) may be the whole of it.

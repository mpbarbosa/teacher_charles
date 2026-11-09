# Teacher Charles — the phonetic method

> **Status: method, not deliverable.** The only part of this that is in the
> product today is the IPA line (`teaching/phonetics.py`). Everything that
> touches the student's **voice** sits behind E9's gate, and **that gate is now
> decided: it opens** — median 0.154 against the proxy's 0.676, the curated
> operation named for **all eight** rules on human speech, and the decision
> written with what it does not decide (plan §3.7; the evidence is in plan §3.5
> and plan §5's E9 row). What follows is the pedagogy
> that E10–E12 would implement, derived from the measurement E8 and E9 already
> ran. One thing in it is code and not pedagogy: `--rescore`, which answers the
> gabarito column again over the reports already on disk, and §7 reports what it
> answered.

`scripts/measure_phonemes.py --corpus [--errado]` was built to answer one
question: *does this stack hear a Brazilian's English?* In answering it, it
assembled every part a pronunciation lesson needs — a curated hypothesis about
the error, a phrase that loads the contrast, two renditions one phoneme apart,
a refusal for audio that is not audio, an alignment that **names the
operation**, and a table that keeps "did not err" apart from "was not
measured".

So this document does not invent a method. It swaps the roles in the one that
is already running: where the procedure has a corpus, the method has a student;
where the procedure has someone reading the table, the method has the evaluator
of E11. **Everything that protects the measurement turns out to protect the
student, and everything that protects the student was free.** That is the whole
argument, and §9 is where it could be shown to be wrong.

*A reference written **plan §X** is to `docs/plano-fase-2.md`; a bare §X is to
this document.*

---

## 1. The unit of teaching is a hypothesis, not a phoneme

`SUBSTITUTIONS` in the measurement script is already a lesson plan, field by
field. Nothing below is new; it is the same tuple read as pedagogy.

| field | in the procedure | in the lesson |
|---|---|---|
| `rule_id` | names the row | what the correction **cites** and what `teaching/schedule.py` schedules |
| `phrase` | what gets recorded | the exercise — and it must load the contrast **more than once** (§4) |
| `right` / `wrong` | documentation | the pair the student **hears** (§2.2) and the instruction for the deliberate pass (§2.4) |
| `swap` | the gabarito | the signature the alignment is intersected with (§2.5) |
| `why` | a comment for whoever reads the script | `explanation`, in English, worded by what the measurement sustains (§4) |

The unit is the **hypothesis**, not the phoneme, for two reasons the
measurement forces. A phoneme has no cause attached, and the cause is the
teachable part: *"Portuguese does not start a syllable with /sp/"* is a lesson,
`/ɪ/` is a symbol. And the identity of a vowel is not reliable in this stack
(plan §2.2 and §3.4) while the **class** of the deviation is — so a unit
defined as "this phoneme" would be a unit the evaluator cannot resolve.

Three fields the measurement showed are missing, and E10 owes them:

* **`signature`** — the *set* of alignment operations that count as this error
  (plan §3.4). Not one operation. §7 measures it: two items fail on one symbol
  and pass as a class, on evidence already on disk, and no correct take matched
  a widened signature.
* **`equivalences`** — what counts as correct English besides the gabarito,
  conditioned on context where it has to be (plan §3.5: `/ɔː/ ≡ /ɑː/`, `/ðɪ/ ≡
  /ðə/`, and `/s/ → /z/` between vowels, which is not a pair but a rule).
* **`discrimination`** — the status of the item against both evidence sources,
  synthetic (`--discriminate`) and human (`--corpus --errado`). §7 is where
  they disagree, and the disagreement decides whether the item is graded or
  only heard.

---

## 2. The cycle: six steps, and where each one comes from

**Where the cycle runs, before what it contains.** The six steps are a block of
their own — the drill — and **one rule per session**, which is plan §4's
exclusion of multi-rule drills and the prefix-cache reason behind it, unchanged.
They are not spliced into a conversational turn: when the student speaks in free
conversation, nothing here runs except the IPA line, the phoneme recognition
happens **after** the turn and in parallel with the teacher's speech, and the
feedback appears when it appears (plan §3.1). That is a pedagogical choice the
measurement made free: *conversation first* is E4's conclusion from Phase 1, and
pronunciation feedback is review material, not interruption. A student
interrupted mid-sentence with phonetics is the failure this ordering avoids,
and it is also the only reason the recognizer's 0.94 s costs nothing.

### 2.1 See the sound — delivered, and outside the gate

The IPA line under every bubble, from espeak, the same G2P that produces the
gabarito and the same one Kokoro speaks with. The student reads the symbol and
hears the sound it stands for.

It is already in the product and it runs for everybody, because it needs no
microphone and no 318 MB weight. The rule the persona partial already states
holds for the whole method: **the line is the pronunciation of the written
words, never a reading of the student.** `prompts/partials/app/phonetics.md`.

### 2.2 Hear the contrast — the `--discriminate` pair, read as a teaching asset

The measurement synthesizes the phrase twice: once from the correct
transcription and once with the curated swap applied **at the phoneme level**,
same voice, same prosody. It did that so the PER difference would be
attributable to the swap and nothing else.

That is also the cleanest minimal pair a pronunciation lesson can get. Two
recordings of *beach* and *bitch* differ in the vowel **and** in everything a
second take of a human voice differs in; these two differ in one phoneme,
by construction.

Two consequences:

* **This step is outside E9's gate.** It plays audio, it does not judge it.
  Nothing here depends on the recognizer working.
* **It survives the blind spots.** An item the evaluator cannot score can still
  be *heard*. §7 proposes that as a status, not as a rejection.

### 2.3 Say it — read the phrase, at a recorded operating point

The student reads the curated phrase. The capture conditions go in the record
with the result, for the reason §3.5 of the plan gives for the corpus: a
measurement whose capture conditions are not written down cannot be re-measured
when a threshold moves. §8.

### 2.4 Say it wrong on purpose — and this is the step that pays for itself

The `--errado` pass exists because in the correct pass *"predicted error not
named"* sums two opposite things: the evaluator missed it, or the error was not
there. Committing the error deliberately breaks the tie.

Read as pedagogy, it breaks a **three-way** tie that one read-aloud conflates —
can the student hear the contrast, can the student produce both sides of it, and
does the student get it right when not thinking about it:

| deliberate error named? | unmonitored error present? | what that is | what to teach |
|---|---|---|---|
| yes | no | full control, and the evaluator is **confirmed** for this student and this rule | graduate |
| yes | yes | hears it, can produce either side, still defaults to the L1 habit | production drill: the habit, not the sound |
| **no** | **yes** | the evaluator does see the rule — it named it in the unmonitored take — and the student cannot aim at it | awareness first: they do not know which of the two they are saying |
| no | no | nothing was named at any point: either the contrast is learned, or it is invisible in this voice, and the pass does not say which | **not a success.** Heard-only (§7) until some take names the operation |

Rows two and three are the same observation without the deliberate column —
*the error showed up* — and they need opposite lessons. That is what the step
buys, and it is the reason it is worth a recording the student does not
otherwise need to make.

The last row is plan §7.2's guard, one layer down. There, *"an analysis that
did not run graduates nobody"*, because a success is the **absence** of a
correction and a broken analyzer produces the same absence. Here the absence
has a second cause — the evaluator may not resolve this rule in this voice at
all — so **a rule whose operation has never once been named for this student
cannot be graduated by silence.** The deliberate pass is what turns that
silence into evidence, and it is the only thing that does.

That is also what makes the step a per-student calibration: a deliberate error
the alignment names is proof that this stack hears this rule in *this* voice —
E9's gate run for one person instead of for the corpus.

Two rules the step needs, and neither is about measurement:

* **Order.** The deliberate pass comes after §2.2, never before: a student who
  has not heard the contrast cannot aim at either side of it.
* **The session ends on the correct production.** The deliberate pass is
  calibration, not drill. It is recorded once per rule and never scheduled
  twice. §10 keeps the risk this leaves open.

### 2.5 The verdict, or silence

Three gates in order, and they are §3.

### 2.6 The schedule

`pronunciation_signal` feeds `teaching/schedule.py` by adding a value to
`rule_review.source` and nothing else — the reconciliation §7.4 of the plan
already wrote. Three clean productions graduate the rule and take it out of the
queue for seven days; the first rung of the ladder is half a day, not zero.

§6 is what pronunciation adds to that schedule that grammar could not give it.

---

## 3. The three gates of one utterance

Each gate answers a different question, and they are not interchangeable. The
layer each one lives in follows the one-way rule of Phase 1.5: `voice/` takes
bytes, `teaching/` takes text.

| | question | layer | what the student sees | what gets recorded |
|---|---|---|---|---|
| 1 | **Is this audio?** | `voice/` | capture advice, in English, about the room and the microphone — never about pronunciation | nothing |
| 2 | **Is this the phrase?** | `teaching/` | "I did not catch that — read it once more" | nothing |
| 3 | **Is there a curated operation in the alignment?** | `teaching/` | the correction, the success, or nothing | the signal |

**Gate 1 is the five questions of plan §3.5** — silence, opening transient,
clipping, energy below 100 Hz, speech contrast — and its refusal text is
already student-facing advice. It must never leak into pronunciation: a student
told they mispronounced a phoneme when the microphone was listening to the room
receives exactly the failure that `POST /api/stt` was designed to avoid.

**Gate 2 is identity, not quality**, and that distinction is what lets it use a
gross aggregate threshold where plan §3.3 forbids one. That argument is that in
human speech the band of correct speech and the band of curated error
**overlap** — 0.364 clean against 0.188 with a real error — so no PER ceiling
separates good pronunciation from bad. It says nothing about separating *the
phrase* from *not the phrase*: the deliberate take at PER 2.500
(`e9-epenthesis-final-errado-d2d3b42e`, 44 phonemes heard against 16 expected)
is not a pronunciation worth scoring, and nothing subtle is needed to see that.
Gate 2 is the floor under the method, and it is also what makes §6 possible.

**Gate 3 is per operation, never per PER.** The alignment is intersected with
the item's signature; only an operation carrying a `rule_id` reaches the
student. §3.2 of the plan, with the number that justifies it.

---

## 4. What the student is told, and in what words

Three verdicts, and there is no fourth:

| | when | what reaches the student |
|---|---|---|
| **correction** | an operation in the alignment matches the item's signature | the error, by name, in the word it happened in |
| **success** | the utterance passed gates 1 and 2, the gabarito says the contrast occurred *n* times, no matching operation appeared, **and the rule has been named at least once for this student** (§2.4) | the rule is confirmed, and the schedule hears about it |
| **silence** | anything else | nothing about pronunciation at all |

### The wording rule, and it is drawn from the floor

§3.4 of the plan says the feedback speaks in classes and not in symbols. The
human data refines that into a rule with a line in it: **name the phoneme where
the measurement resolves identity, name the class where it does not.**

* **Consonant identity is resolved.** `/θ/→/t/`, `/ð/→/d/`, `del /h/` and
  `/ŋ/→/n/` were named on human speech, in the right word — `/ɹ/→/h/` so far
  only on the synthetic pair. Saying *"the `<th>` in* other *came out as /d/"*
  is supported.
* **Vowel identity is not.** The floor of plan §2.2 is built of rhotic and reduced
  vowel, and §7 below shows the two vowel items landing one symbol away from
  their own gabarito on real speech. Saying *"you said /ɐ/ where /ɪ/ was
  expected"* is precision the stack does not have.

Which gives, for the eight curated items, what the student would hear:

| rule | what reaches the student |
|---|---|
| `pron.th-unvoiced` | "In *think*, the `<th>` came out as /t/. In English it is a breath between tongue and teeth — nothing touches." |
| `pron.th-voiced` | "In *other*, the `<th>` came out as /d/. The other three in that sentence were right — *they*, *the*, *brother*. It is the same sound in all four." |
| `pron.epenthesis-initial` | "You added a vowel before the /sp/ — it came out *isport*. English begins the word on the /s/." |
| `pron.epenthesis-final` | "You added a vowel after *big*. English ends the word on the consonant, with no extra beat." |
| `pron.vowel-length` | "The vowel in *beach* came out short. English holds that one long, and the short one is a different word." |
| `pron.h-dropping` | "Three of the four /h/ in that sentence disappeared. Portuguese `<h>` is silent; English /h/ is a real breath." |
| `pron.ing-alveolar` | nothing — see §7 |
| `pron.r-initial` | "At the start of *red*, the /ɹ/ came out as the Portuguese `<r>`. English /ɹ/ is made with the tongue, not in the throat." |

Counting the opportunities is what makes the second and sixth lines possible,
and that is a curation requirement, not a wording choice: the phrase of
`pron.th-voiced` carries `/ð/` four times and `pron.h-dropping` carries `/h/`
four. **A phrase with one opportunity cannot say "three of the four were
right", and cannot tell a clean production from a lucky alignment.** Hence the
rule in §1: the phrase loads the contrast more than once.

### What is never said

* **No score, no percentage, no 0–100.** §4 of the plan, "Fora,
  explicitamente": the stack sustains a class of error, not a fine measure. A
  number would be invented precision.
* **Nothing about prosody, stress or rhythm.** The alignment does not see them.
* **No claim by Charles that he heard the student.** The persona partial
  already forbids it, and this method does not change it: the verdict comes from
  the evaluator, as structured data with a `rule_id`, exactly as a grammar
  correction does.

---

## 5. The method on one page: `pron.th-voiced`, two real takes

The phrase is *"They brought the other brother"*, and it carries `/ð/` four
times. Both takes below are clean — they passed the five questions, with no
edge warning — and both reports are on disk, though only the later WAV is:
the recording of the same rule overwrote the earlier one, and the alignment
needs only `esperado` and `ouvido`.

| | `e9-th-voiced-b704dc72` | `e9-th-voiced-3ba35169` |
|---|---|---|
| PER | 0.188 | 0.125 |
| what the alignment held | `/ɔː/→/oʊ/`, `/ɪ/→/ɚ/`, **`/ð/→/d/` in *other*** | `/ɔː/→/ɑː/`, `/ɪ/→/ə/`, all four `/ð/` right |
| matched the signature? | yes | no |
| the verdict | **correction**, citing `pron.th-voiced` | **success**, over four opportunities |
| what the student hears about the rest | nothing | nothing |

The take with the higher PER is the one with the error, and that is a
coincidence: plan §3.5 measures a clean take at 0.364 and this erroneous one at
0.188. **The verdict is not reading the PER column.** It reads the operation,
and the two divergences in the right-hand take — the *cot/caught* merger and
the reduced article — are correct English that the gabarito does not spell. If
they reached the student, the method would be correcting an error nobody made,
which is the worst outcome available (§6 of the plan, and the reason
`POST /api/stt` records no turn).

---

## 6. Pronunciation pays the debt the grammar drill left open

§7.2 of the plan states the asymmetry: the analyzer returns errors, so a
success can only be inferred as *the absence of an error where the error would
have shown up*, and the drill is the only place that builds the opportunity.
The plan's §6 then records the debt that leaves: inside the drill, **"did not err" and
"did not try" are the same observation**, and a student answering *"yes"* three
times graduates the rule.

A read-aloud closes that, and it is the first signal in the product where the
opportunity is checkable by machine rather than assumed:

* **The gabarito says the opportunity existed, and how many times.** The
  phonemized phrase contains `/ð/` four times whether or not the student says
  anything.
* **Gate 2 says the student tried.** An utterance that is not the phrase is
  *"did not try"*, and it records nothing — it does not become a success by
  being free of the error. The grammar drill has no equivalent of that gate,
  because any text is a plausible answer to a question.
* **So the signal row carries `opportunities` and `errors`, not a boolean.**
  Four `/ð/` right is stronger evidence than one, and the scheduler can ask for
  more than mere absence before it graduates a rule.

What this does **not** fix is the same thing read-aloud never fixes: see §10.

---

## 7. The curation protocol for `pronunciation.yaml`, and what the reports already decide

E10's rule from §3.2 of the plan is that no item enters the base without
surviving `--discriminate`, and plan §3.7 adds the second half: the operation has
to be named **by human speech, in a take the diagnostic did not warn about**.
Three rules pass that today. The human passes are therefore a second source of
evidence, and where the two disagree the item is not rejected — it is **sorted**:

| status | means | what the student gets |
|---|---|---|
| **graded** | the operation, or the signature set, was named on a **clean** human take | all six steps of §2 |
| **named, not clean** | the operation was named, only on a take the diagnostic warned about — five of the eight | the §2 steps that need no verdict, until a clean take arrives |
| **heard only** | the contrast is real and documented, the alignment does not resolve it | §2.1, §2.2 and §2.3 — never a verdict |
| **rejected** | no evidence the deviation exists, or the phrase does not load it | not in the base |

The second row is the one this corpus mostly produces, and it is a statement
about the **room** rather than about the stack: fourteen usable takes out of
sixty-four, and ten of the eleven refusals in the deliberate pass are speech
contrast. A rule in that row is one quiet recording away from the first.

### What the reports on disk say, read item by item

The table below is read off `gravacoes/*.json` — the correct pass recorded in
§3.5 of the plan, and the deliberate pass of the same corpus. **It is my
reading of those reports, not a line the plan has already written**, and three
of its rows disagree with what the synthetic measurement of §2.3 concluded.

Only three deliberate takes are clean by the criterion E9 counts with
(`th-voiced`, `h-dropping`, `ing-alveolar`); the ones marked *warned* were
recorded below the level the diagnostic wants — two of them also with the head
cut, and in both the matched operation sits downstream of it — and the plan's
rule is that a warned report does not count. They are in the table because their alignments
are informative about the **class** of what was heard, which is the question
plan §3.4 asks — not because they close anything.

| rule | unmonitored | deliberate | named? | the cell of §2.4 |
|---|---|---|---|---|
| `pron.th-unvoiced` | 0.053 twice, `/θ/` right | 0.105 and 0.789, both warned, `/θ/→/t/`; a third take came out `/θ/` right | **yes** | control |
| `pron.th-voiced` | 0.188 with `/ð/→/d/`; 0.125 all four right | 0.438, `/ð/→/d/` ×3 | **yes** | habit |
| `pron.h-dropping` | 0.364, all four `/h/` produced | 0.818, three of the four `/h/` gone | **yes** | control |
| `pron.epenthesis-initial` | 0.312, no epenthesis | three warned takes: 0.438 **`+/i/`**, 0.438 **`+/ɪ/` ×2**, 0.250 **`+/ɪ/` ×2** — the gabarito's own symbol | **yes**, and by class | control |
| `pron.vowel-length` | 0.059, both `/iː/` long | 0.706 (warned), **`/iː/→/i/` twice**; a later take matched the class at PER 1.000, which is an accident | **by class** | control |
| `pron.ing-alveolar` | 0.529, **`/ŋ/→/n/` twice** | 0.235, `/ŋ/` produced in both | no | **awareness** |
| `pron.epenthesis-final` | 0.312, `/ɡ/→/iː/` in place of the epenthesis | five takes, none clean: 0.250 (warned) with **`+/i/` and `+/ɪ/`**; before it two refused and two warned (2.500, 0.875) | **yes**, and by class | control |
| `pron.r-initial` | **0.000**, and 0.154 | 0.615 and 0.308, both warned, **`/ɹ/→/h/` named on each**; one earlier take refused at 1.8× | **yes**, twice | control |

**What `--rescore` answers over the 64 reports on disk**, and the column that
decides anything is the last one:

| pass | reports | with a gabarito | named | clean | named among the clean |
|---|---|---|---|---|---|
| unmonitored | 37 | 30 | 4, all exact | 11 | **2** |
| deliberate | 27 | 16 | 13 — 10 exact, **3 by class** | 3 | **2** |

The deliberate row keeps growing — the recording session was still running as
this was written — and the two columns on the right have not moved through any of
it. That is the point of reading them: `--rescore` re-answers the whole table
with no microphone and no model, and only a clean take changes what the criterion
sees.

The widening added two named errors and moved **neither** gate-relevant number:
both class matches sit on takes the diagnostic warned about, and a warned report
does not count. The two items are therefore not closed — they are one clean
deliberate take away, which is the cheapest thing left in E9.

Five findings. The first two have been applied and re-scored, and the fourth is
what applying them exposed.

**Epenthesis is named by class, and it took no new recording.** The deliberate
take `e9-epenthesis-initial-errado-454a1ff6` heard
`/i s p ɔ ɾ t … f ɔ i s t ɹ ɛ s/`: the inserted vowel is audible before **both**
`/sp/` and `/st/`, though the alignment books only the first as an insertion and
charges the second to the vowel of *for* next door — plan §3.5 reads that as an
attribution caveat for E11, and it is why the student is shown the rule and
never a count of occurrences. The gabarito asks for `ins /ɪ/` because espeak renders
the swap that way; the recognizer heard `ins /i/`, which is arguably the better
transcription of what a Brazilian inserts. §3.4 of the plan prescribes the fix
— the signature is a *set* — and `scripts/measure_phonemes.py` now applies it:
`widen()` opens the **heard side** of each gabarito operation to the class
`{ɪ, i, ɐ, ə}`, and `--rescore` answers the column again over the reports on
disk, with no microphone and without loading the 318 MB recognizer. The take
now reads `classe`, naming `ins /i/`. The synthetic pass missed this item for
the same reason (`+/ɐ/` against `+/ɪ/`), so one class fixes both sources.

**`pron.vowel-length` is the same story, in substitution**, and it re-scored
the same way. The deliberate take heard `/iː/→/i/` in both *beach* and *people*
where the gabarito says `/iː/→/ɪ/`. The class — the long vowel came out short —
is what the item describes, and the row now reads `classe`, naming
`sub /iː/→/i/`.

Three properties of that widening, and the third is the one that matters:

* **It changes nothing about correctness.** Only which operations count as the
  **curated error** move. The alignment, the PER and what counts as correct
  English are untouched — the equivalence sets of §3.5 of the plan
  (`/ɔː/ ≡ /ɑː/`) are a different table, for a different question, and E10's.
* **It is kept out of `run_pairs`.** The floor, signal and accent numbers of §2
  of the plan were measured with the strict intersection; widening there would
  redefine those tables in silence.
* **It did not misfire.** Across the 37 reports of the correct pass — 30 with a
  gabarito, 11 of them clean — **no take matched by class at all.** That is the
  falsification test §9 names for this change, run on the only speech there is,
  and it came back clean.

**`pron.ing-alveolar` inverts, and the synthetic blind spot was synthetic.**
§2.3 of the plan reports the recognizer hearing `/ŋ/` on both sides of the
Kokoro pair and names the item a blind spot. On human speech it did the
opposite: the unmonitored take came out `/k ə n/` and `/d ɪ n/` — alveolar
twice, named twice — and the *deliberate* take produced `/ŋ/` in both endings.
So the recognizer distinguishes `/ŋ/` from `/n/` in this voice; it was Kokoro's
rendition of the swap that it could not. §3.5 of the plan called the 0.529 take
the noisiest clean utterance in the set and the naming therefore uncertain; the
deliberate pass does not confirm it, but it does not refute it either, because
what failed in the deliberate pass may have been the speaker. For the method the
cell is what matters, and the cell is *awareness*: this speaker produces the
alveolar when not watching and the velar when asked for the alveolar. That is a
lesson, and it is not the lesson the other seven items get.

**Applying it exposed a false positive that was already there, and a second one
followed.** The deliberate take `e9-epenthesis-final-errado-d2d3b42e` — 44
phonemes of salad against 16 expected, PER 2.500, three warnings — matched
`ins /i/` and read `sim`; a later `pron.vowel-length` take matched the class at
PER 1.000, where nothing aligned at all. In an
alignment that wrong a one-operation gabarito matches by accident: the "sim"
looks like a measurement and is a coin toss. The aggregate column had been
mixing the countable takes with the excluded ones since it existed, which is the
same confusion the median fix of §3.5 of the plan corrected one column over. The
table now also prints the named fraction **over the clean takes**, and that is
the line to read — 2 of 3 in the deliberate pass, 2 of 11 in the unmonitored
one. The method inherits the rule directly: **a verdict never comes from a take
the diagnostic warned about**, however well the operation matches.

**And the unmonitored pass has one of its own, which names the vulnerable shape
of gabarito.** `e9-h-dropping-13008743` transcribes as six phonemes against
eleven — the take ends after *He has a* — and `del /h/` matched because the last
two `/h/` are not there to be dropped. A take that stops early deletes
everything after it, so **a deletion signature is the one shape that a truncated
take can satisfy for free**, and h-dropping is the only curated rule whose
signature is a deletion. For E11 that is not a new gate — **gate 1 had already
named the cause**: the second of this take's two warnings is *"a fala vai até
6.9s de 7.0s, encostada no fim do arquivo: o último fonema pode ter ficado
fora"*. The diagnostic said the end was cut, and the verdict would have been
suppressed before the gabarito was consulted. What it does ask of E10 is that a
deletion item be curated knowing this, and the clean take of the same phrase is
the reassurance: all four `/h/` produced, and "não".

**Most takes are not usable, and that is a product constraint, not a corpus
detail.** The deliberate pass recorded 27 takes at the operating point of plan §3.5:
**11 refused**, ten of them on speech contrast (1.8× to 3.3× against a minimum
of 4), **13 warned**, **3 clean**. One usable judgment in eight attempts is a price a corpus
can pay and a student cannot. Two consequences for E12, and both are cheap:

* **Gate 1 runs before the student reads, not after.** The five questions need
  no model and no phrase — a second of room noise answers the contrast question
  on its own. Refusing *after* someone has read a sentence spends their effort
  to tell them about their room.
* **The lesson does not depend on the microphone.** §2.1 and §2.2 carry
  content with no capture at all, which is what keeps a method alive in a room
  that fails the fifth question.

### The requirements an item carries

1. `phrase` loading the contrast **at least twice** (§4), and short enough to
   read in one breath — the measured takes are 11 to 19 phonemes.
2. `signature`: the set of acceptable operations, from plan §3.4 and widened by the
   evidence above.
3. `equivalences`: what correct English also does here. Three are known and the
   third needs context, not a pair: `/ɔː/ ≡ /ɑː/`, `/ðɪ/ ≡ /ðə/`, and
   `/s/ → /z/` between vowels — which appeared in *this is* in three separate
   takes, including one deliberate one, and is therefore systematic in this
   speaker rather than a lapse.
4. `discrimination`: the status from both sources — one of the four in the table
   above — with the md5 of the takes that decided it, and the warning each one
   carried if it carried any. A status that does not name its take cannot be
   re-checked when a threshold moves, which is §8's whole point.
5. `explanation`, in English, worded by §4's line between phoneme and class.

---

## 8. The operating point is part of the data

§3.5 of the plan records the capture conditions of the corpus because a corpus
whose conditions are unwritten cannot be re-measured when a threshold moves.
`SPEECH_CONTRAST_MIN = 4` is the standing example: it was calibrated by
re-recording the same phrase in a quieter room, not by lowering the threshold,
and that only worked because the gain was written down and held.

A student's pronunciation history is in the same position, for the same reason.
So `pronunciation_signal` stores, with each row, what gate 1 measured — speech
contrast, peak, the speech span, the phoneme count — alongside the operation
and the `rule_id`. It costs a handful of floats, and it buys the thing plan §3.5
needed twice: when the threshold changes, the history can be re-read instead of
re-recorded. The student cannot be asked to say it all again.

---

## 9. How this method could be shown to be wrong

Each claim with the measurement that would break it. Three of the five can be
run on the corpus that exists.

| claim | breaks if |
|---|---|
| The deliberate pass separates "the evaluator is blind" from "the error was absent" | the named fraction stays at the correct pass's level. **Re-scored: 13 of 16 measured deliberate takes against 4 of 30 unmonitored, and 2 of 3 clean against 2 of 11 clean. The separation holds on both bases, and the clean one rests on three takes** |
| Widening the signature to a class grades epenthesis and vowel length without false positives | a clean, correct take matches a widened signature. `/ɐ/` is one of the three phonemes the floor of plan §2.2 is made of, so `ins /ɐ/` is the term most likely to misfire. **Run: zero class matches across the 30 gabarito-carrying takes of the correct pass, 11 of them clean** |
| Consonant identity can be named and vowel identity cannot | a vowel item gets named by symbol on human speech across replicas, or a consonant naming fails to replicate |
| Gate 2 separates "did not try" from "did not err" | a read-aloud of something else passes gate 2 and records a success |
| A clean read-aloud is evidence about the student's pronunciation **in conversation** | it is not tested at all, and §10 says so first |

---

## 10. The debts this method opens

* **Read-aloud is not conversation, and this whole method is read-aloud
  shaped.** A student reading a curated phrase monitors their own speech; the
  same student talking to Charles does not. Everything above measures the
  monitored production, and the table of §2.4 depends on the unmonitored
  side being a *reading* too, not speech. Nothing here evaluates pronunciation
  in a free turn, and nothing here should be presented as doing so.
* **Rehearsing the error.** The deliberate pass asks a student to produce the
  thing they are trying to lose. The guards in §2.4 — after the contrast, once
  per rule, never last in the session — are reasoning, not measurement. If a
  rule's error rate rises after its deliberate take, the step is costing more
  than it buys, and `rule_review` already holds the history to see that.
* **One speaker's variants are not the language's.** The three equivalence
  sets come from eleven clean takes of one Brazilian voice. A second speaker
  may make `/s/ → /z/` an error rather than a variant, and the method would then
  be forgiving a real deviation. Equivalences are an item field for that
  reason, not a global table.
* **Five of the eight rules are named only on a take the criterion excludes.**
  `th-voiced`, `h-dropping` and `ing-alveolar` have a clean deliberate take; the
  other five — `th-unvoiced`, `epenthesis-initial`, `epenthesis-final`,
  `vowel-length`, `r-initial` — name their operation on a warned one, and in
  every case the warning is about the recording and not the transcription. Two
  of the five earned their row late and cheaply: `r-initial` names `/ɹ/→/h/`
  twice over the cleanest unmonitored take in the set (0.000), and
  `epenthesis-final` names `+/i/` and `+/ɪ/` in one take after three unusable
  ones. What they owe is a quiet room, which is the finding below, not a
  different stack.
* **No prosody, no rhythm, no stress.** §4 of the plan, unchanged. The
  alignment is of phonemes, and the student who gets every phoneme right can
  still be unintelligible for reasons this method cannot see.
* **The perception step is unmeasured.** §2.2 is the cheapest step here and the
  only one with no number behind it: nobody has measured whether hearing the
  synthetic contrast changes what the student then produces. It costs nothing to
  ship and it is not therefore known to work.

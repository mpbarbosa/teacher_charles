-- Phase 2, E12: pronunciation in the product, read aloud.
--
-- The drill of docs/metodo-fonetico.md §2 as a surface: the student hears the
-- contrast pair, reads one curated phrase three times, and the session gives
-- the schedule ONE review from the tally (plan §5, read-aloud paragraphs).
-- Three things go in, and the third is a rebuild:
--
--   `pronunciation_session` -- one read-aloud session over one rule.
--   `pronunciation_signal`  -- one judged take, per metodo §3's table: a take
--                              silenced at gate 1 or 2 records NOTHING, so a
--                              row here is always a take that reached gate 3.
--   `rule_review`           -- rebuilt, because a pronunciation review has no
--                              turn and SQLite cannot relax a NOT NULL or a
--                              CHECK in place.

CREATE TABLE pronunciation_session (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER NOT NULL REFERENCES student(id) ON DELETE CASCADE,
    -- A graded rule of data/contrastive/pronunciation.yaml.
    rule_id     TEXT    NOT NULL,
    -- Which of the rule's phrases this session reads. One per session: the
    -- tally counts takes of the SAME phrase, or "2 of 3" means nothing.
    phrase      INTEGER NOT NULL DEFAULT 0,
    -- Who chose the rule. Kept because a schedule that keeps picking a rule
    -- the student never asks for is a schedule to look at.
    picked_by   TEXT    NOT NULL CHECK (picked_by IN ('student', 'schedule')),
    -- NULL while open. Set once, by the one statement that also claims the
    -- review (`WHERE ended_at IS NULL`), which is what keeps a double click
    -- on "finish" from reviewing twice on the shared connection.
    ended_at    TEXT,
    -- What the tally decided: 'miss' and 'hit' reviewed the rule; 'once' and
    -- 'none' did not, and that is information, not a missing value.
    outcome     TEXT    CHECK (outcome IN ('miss', 'hit', 'once', 'none')),
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_pronunciation_session_student
    ON pronunciation_session(student_id, rule_id, id DESC);

CREATE TABLE pronunciation_signal (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES pronunciation_session(id) ON DELETE CASCADE,
    student_id    INTEGER NOT NULL REFERENCES student(id) ON DELETE CASCADE,
    rule_id       TEXT    NOT NULL,
    phrase        INTEGER NOT NULL,
    -- metodo §2.4: 'deliberate' is the take said wrong on purpose, once per
    -- rule and never scheduled; 'unmonitored' is a take of the drill proper.
    -- Only the second kind is tallied; the first is what lets a later clean
    -- read count, because it proves this stack hears the rule in this voice.
    pass          TEXT    NOT NULL CHECK (pass IN ('unmonitored', 'deliberate')),
    -- Never 'silence': a silenced take is not a signal (see the header).
    verdict       TEXT    NOT NULL CHECK (verdict IN ('correction', 'no-error')),
    -- Plan §5's two numbers per take: how many times the phrase gave the
    -- contrast a chance, and how many curated operations the alignment named.
    -- No `errors <= opportunities`: the signature is widened to the target's
    -- class (teaching/pronunciation.py::widen), so a named operation need not
    -- sit on the target symbol itself, and the bound would turn a judged take
    -- into a failed insert.
    opportunities INTEGER NOT NULL CHECK (opportunities > 0),
    errors        INTEGER NOT NULL CHECK (errors >= 0),
    -- The named findings as JSON, [{"word", "expected", "heard"}]: what the
    -- card says "in other ... 2 of 3 takes" from. Never the transcription.
    findings      TEXT    NOT NULL DEFAULT '[]',
    -- The operating point (metodo §8): a measurement whose capture is not
    -- written down cannot be re-measured when a threshold moves.
    model         TEXT    NOT NULL,
    duration_s    REAL    NOT NULL,
    speech_from_s REAL,
    speech_to_s   REAL,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_pronunciation_signal_session ON pronunciation_signal(session_id);
CREATE INDEX idx_pronunciation_signal_rule ON pronunciation_signal(student_id, rule_id);

-- `rule_review`, rebuilt. Same columns and the same rows, with three changes:
--
--   * `source` admits 'pronunciation' (plan §7.4: the second source hangs on
--     the schedule by adding a value here and nothing else);
--   * `turn_id` may be NULL, because a read-aloud session has no turn;
--   * `pronunciation_session_id` names the session instead, and EXACTLY one
--     of the two is set -- a review always points at the evidence it came
--     from, which is what `turn_id` promised in 004.
--
-- The unique index is "one review per session" held by the schema and not
-- only by the route.
CREATE TABLE rule_review_new (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id               INTEGER NOT NULL REFERENCES student(id) ON DELETE CASCADE,
    rule_id                  TEXT    NOT NULL,
    turn_id                  INTEGER REFERENCES turn(id) ON DELETE CASCADE,
    pronunciation_session_id INTEGER REFERENCES pronunciation_session(id) ON DELETE CASCADE,
    hit                      INTEGER NOT NULL CHECK (hit IN (0, 1)),
    source                   TEXT    NOT NULL
                             CHECK (source IN ('drill', 'lapse', 'pronunciation')),
    box_before               INTEGER NOT NULL,
    box_after                INTEGER NOT NULL,
    interval_days            REAL    NOT NULL,
    created_at               TEXT    NOT NULL DEFAULT (datetime('now')),
    CHECK ((turn_id IS NULL) <> (pronunciation_session_id IS NULL)),
    CHECK ((source = 'pronunciation') = (pronunciation_session_id IS NOT NULL))
);

INSERT INTO rule_review_new (
    id, student_id, rule_id, turn_id, hit, source,
    box_before, box_after, interval_days, created_at
)
SELECT id, student_id, rule_id, turn_id, hit, source,
       box_before, box_after, interval_days, created_at
  FROM rule_review;

DROP TABLE rule_review;
ALTER TABLE rule_review_new RENAME TO rule_review;

CREATE INDEX idx_rule_review_student ON rule_review(student_id, rule_id, id DESC);
CREATE UNIQUE INDEX idx_rule_review_pronunciation_session
    ON rule_review(pronunciation_session_id)
    WHERE pronunciation_session_id IS NOT NULL;

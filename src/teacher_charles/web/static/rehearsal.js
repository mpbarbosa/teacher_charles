"use strict";

// The rehearsal beside a lesson's full text: saying the text aloud, block by
// block, against the clock.
//
// Loaded whenever the page has a copy of the text, voice or no voice: what it
// needs is the browser's own clock and microphone. The blocks are the copy's
// paragraphs and the template draws their heads (lessons.html); what the
// lesson knows about each one -- its title and target time -- comes from the
// YAML's `rehearsal`, and the cue words from the copy (teaching/lessons.py,
// RehearsalBlock says why they are not in the YAML).
//
// ## What stays where
//
// Nothing here talks to the server. The pace, the last five times of each
// block and the last run-throughs are kept in this browser's localStorage,
// per lesson, as a convenience: a rehearsal time is not evidence the app
// teaches from -- no level, no schedule, no correction reads it -- so it does
// not go into the student's database, for the reason /api/syntax records
// nothing. Every read and write is guarded: a private window with no storage
// rehearses just the same and forgets on reload.
//
// A recording never leaves the page. It is a blob in memory, played back
// through an object URL, and it dies with the page. It does not go through
// /api/stt either: the word takes beside the vocabulary do (lesson.js), but a
// block is a minute of speech, and Whisper's reading of it would be a
// transcript dressed as a check, which is not what this is.
//
// ## The voice
//
// "Hear it" belongs to the reader (lesson.js, `button.read-block`), so a block
// is heard in the same voice, lit sentence by sentence, as the full text. This
// script only asks the reader to fall silent -- a `tc:silence` event -- when
// a run-through or a recording starts: the loudspeaker would otherwise speak
// into the microphone and over the student.

(() => {
  const root = document.getElementById("rehearsal");
  const notes = document.querySelector(".rehearsal-notes");
  const toggle = document.querySelector("button.rehearse");
  const reading = document.getElementById("full-text-reading");
  if (!root || !notes || !toggle || !reading) return;

  const blocks = [...reading.querySelectorAll(".rehearsal-block")].map((head) => ({
    head,
    paragraph: head.nextElementSibling, // the <p> the head sits before
    index: Number(head.dataset.block),
    title: head.dataset.title,
    words: Number(head.dataset.words),
    target: head.dataset.target ? Number(head.dataset.target) : 0,
    cues: [...head.querySelectorAll(".block-cues .cue")].map((c) => c.textContent),
  }));
  // The run-through goes through the timed blocks; with none timed, all of them.
  const timed = blocks.filter((b) => b.target);
  const runBlocks = timed.length ? timed : blocks;
  const totalTarget = Number(root.dataset.target) || 0;

  // The sticky part (bar, run panel) and the notes under it (help, history).
  const $ = (selector) => root.querySelector(selector) ?? notes.querySelector(selector);
  const paceInput = $(".pace-input");
  const cuesOnly = $(".cues-only");

  // How many of each block's stopwatch times are kept, and of the run-throughs.
  const KEPT_TIMES = 5;
  const KEPT_RUNS = 15;
  // A recording left running by a forgotten click stops by itself.
  const MAX_TAKE_MS = 5 * 60 * 1000;
  // Opus in webm (Chromium) or ogg (Firefox), the list lesson.js uses.
  const MIME_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/ogg"];

  // --- storage, per lesson ------------------------------------------------- //

  const STORE = `tc.rehearsal.${root.dataset.slug}`;
  const state = (() => {
    let saved = {};
    try {
      saved = JSON.parse(localStorage.getItem(STORE)) ?? {};
    } catch {
      /* no storage, or not ours: start clean */
    }
    return {
      on: saved.on === true,
      wpm: Number(saved.wpm) || Number(paceInput.value),
      times: saved.times && typeof saved.times === "object" ? saved.times : {},
      runs: Array.isArray(saved.runs) ? saved.runs : [],
    };
  })();

  function save() {
    try {
      localStorage.setItem(STORE, JSON.stringify(state));
    } catch {
      /* the times live until the page closes */
    }
  }

  // --- numbers ------------------------------------------------------------- //

  /** Seconds as m:ss. */
  function clock(seconds) {
    const s = Math.max(0, Math.round(seconds));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  /** Seconds a block takes at the chosen pace. */
  function estimate(words) {
    return (words / state.wpm) * 60;
  }

  /**
   * How a time sits against its target: within 5% is on time, within 20% is
   * close, past that is over. The same bands the rehearsal page this was
   * ported from used; under the target is never a fault -- a pitch can end
   * early, and the student sees the number anyway.
   */
  function band(seconds, target) {
    if (!target) return "";
    const ratio = seconds / target;
    return ratio <= 1.05 ? "on-time" : ratio <= 1.2 ? "close" : "over";
  }

  function setTime(el, seconds, target) {
    el.textContent = clock(seconds);
    el.classList.remove("on-time", "close", "over");
    const kind = band(seconds, target);
    if (kind) el.classList.add(kind);
  }

  // --- the estimates and the stopwatch times ------------------------------- //

  function refresh() {
    $(".pace-value").textContent = `${state.wpm} wpm`;
    let totalEstimate = 0;
    let totalTimed = 0;
    let allTimed = true;
    for (const b of blocks) {
      const seconds = estimate(b.words);
      setTime(b.head.querySelector(".est"), seconds, b.target);
      const laps = state.times[b.index] ?? [];
      const lapsEl = b.head.querySelector(".laps");
      lapsEl.replaceChildren();
      if (laps.length) {
        const last = document.createElement("b");
        setTime(last, laps[laps.length - 1], b.target);
        lapsEl.append("last ", last);
        if (laps.length > 1) lapsEl.append(` · before: ${laps.slice(0, -1).map(clock).join(", ")}`);
      } else {
        lapsEl.textContent = "not timed yet";
      }
      if (!b.target && timed.length) continue; // untimed blocks are out of the total
      totalEstimate += seconds;
      if (laps.length) totalTimed += laps[laps.length - 1];
      else allTimed = false;
    }
    setTime($(".total-est"), totalEstimate, totalTarget);
    const totalEl = $(".total-timed");
    if (totalTimed) {
      setTime(totalEl, totalTimed, allTimed ? totalTarget : 0);
      if (!allTimed) totalEl.textContent += " (some blocks untimed)";
    } else {
      totalEl.textContent = "–";
      totalEl.classList.remove("on-time", "close", "over");
    }
  }

  paceInput.value = state.wpm;
  paceInput.addEventListener("input", () => {
    state.wpm = Number(paceInput.value);
    refresh();
    save();
  });

  // One stopwatch at a time: two running would mean two blocks said at once.
  let watch = null; // { block, button, startedAt, timer }

  function stopWatch(keep) {
    if (!watch) return;
    const { block, button, startedAt, timer } = watch;
    watch = null;
    clearInterval(timer);
    button.textContent = "⏱ Time it";
    button.setAttribute("aria-pressed", "false");
    const seconds = Math.round((performance.now() - startedAt) / 1000);
    block.head.querySelector(".block-clock").textContent = clock(seconds);
    if (keep && seconds > 0) {
      const laps = [...(state.times[block.index] ?? []), seconds].slice(-KEPT_TIMES);
      state.times[block.index] = laps;
      save();
    }
    refresh();
  }

  function timeBlock(block, button) {
    const same = watch?.block === block;
    stopWatch(true);
    if (same) return;
    endRun(false);
    const shown = block.head.querySelector(".block-clock");
    const startedAt = performance.now();
    watch = {
      block,
      button,
      startedAt,
      timer: setInterval(() => {
        shown.textContent = clock((performance.now() - startedAt) / 1000);
      }, 200),
    };
    shown.textContent = "0:00";
    button.textContent = "■ Stop";
    button.setAttribute("aria-pressed", "true");
  }

  // --- hiding the text ----------------------------------------------------- //

  function applyCues() {
    reading.classList.toggle("cues-only", state.on && cuesOnly.checked);
  }
  cuesOnly.addEventListener("change", applyCues);

  // --- the run-through ----------------------------------------------------- //

  const panel = $(".run-panel");
  let run = null; // { i, startedAt, lapAt, laps, timer }

  function showRunBlock() {
    const b = runBlocks[run.i];
    $(".run-index").textContent = String(run.i + 1);
    $(".run-count").textContent = String(runBlocks.length);
    $(".run-name").textContent = b.target ? `${b.title} · target ${clock(b.target)}` : b.title;
    const cues = $(".run-cues");
    cues.replaceChildren(
      ...b.cues.map((text) => {
        const span = document.createElement("span");
        span.className = "cue";
        span.textContent = text;
        return span;
      })
    );
    for (const x of blocks) {
      const now = x === b;
      x.head.classList.toggle("current", now);
      x.paragraph?.classList.toggle("current", now);
    }
    // Under the sticky bar and panel, not behind them.
    b.head.style.scrollMarginTop = `${root.offsetHeight + 8}px`;
    b.head.scrollIntoView({ block: "start", behavior: "smooth" });
  }

  function startRun() {
    stopWatch(true);
    silence();
    const now = performance.now();
    run = {
      i: 0,
      startedAt: now,
      lapAt: now,
      laps: [],
      timer: setInterval(() => {
        $(".run-clock").textContent = clock((performance.now() - run.startedAt) / 1000);
      }, 200),
    };
    $(".run-clock").textContent = "0:00";
    panel.hidden = false;
    reading.classList.add("running");
    showRunBlock();
    $(".run-next").focus({ preventScroll: true });
  }

  function nextBlock() {
    if (!run) return;
    const now = performance.now();
    run.laps.push(Math.round((now - run.lapAt) / 1000));
    run.lapAt = now;
    if (run.i < runBlocks.length - 1) {
      run.i += 1;
      showRunBlock();
    } else {
      endRun(true);
    }
  }

  /** Ends the run-through; `finished` keeps it in the history, Esc does not. */
  function endRun(finished) {
    if (!run) return;
    clearInterval(run.timer);
    if (finished) {
      state.runs = [
        ...state.runs,
        {
          when: new Date().toISOString(),
          laps: run.laps,
          total: run.laps.reduce((a, b) => a + b, 0),
          wpm: state.wpm,
        },
      ].slice(-KEPT_RUNS);
      save();
      drawRuns();
      $(".run-history").open = true;
    }
    run = null;
    panel.hidden = true;
    reading.classList.remove("running");
    for (const b of blocks) {
      b.head.classList.remove("current");
      b.paragraph?.classList.remove("current");
    }
  }

  function cell(tag, text, kind) {
    const el = document.createElement(tag);
    el.textContent = text;
    if (kind) el.className = kind;
    return el;
  }

  function drawRuns() {
    const table = $(".runs");
    table.replaceChildren();
    if (!state.runs.length) {
      const row = document.createElement("tr");
      row.append(cell("td", "No run-through yet. Press ▶ Run-through: the time of each block lands here."));
      table.append(row);
      $(".runs-clear").hidden = true;
      return;
    }
    $(".runs-clear").hidden = false;
    const head = document.createElement("tr");
    head.append(cell("th", "When"));
    for (const b of runBlocks) head.append(cell("th", b.target ? `${b.title} (${clock(b.target)})` : b.title));
    head.append(cell("th", totalTarget ? `Total (${clock(totalTarget)})` : "Total"));
    table.append(head);
    for (const r of [...state.runs].reverse()) {
      const row = document.createElement("tr");
      const when = new Date(r.when);
      row.append(
        cell("td", Number.isNaN(when.getTime()) ? "–" : when.toLocaleString([], {
          dateStyle: "short",
          timeStyle: "short",
        }))
      );
      runBlocks.forEach((b, i) => {
        const lap = r.laps?.[i];
        row.append(cell("td", lap == null ? "–" : clock(lap), lap == null ? "" : band(lap, b.target)));
      });
      const total = cell("td", "", band(r.total, totalTarget));
      total.append(cell("b", clock(r.total)));
      row.append(total);
      table.append(row);
    }
  }

  $(".run-start").addEventListener("click", startRun);
  $(".run-next").addEventListener("click", nextBlock);
  $(".run-stop").addEventListener("click", () => endRun(false));
  $(".runs-clear").addEventListener("click", () => {
    if (!state.runs.length) return;
    if (!window.confirm(`Clear the ${state.runs.length} run-throughs kept for this lesson?`)) return;
    state.runs = [];
    save();
    drawRuns();
  });

  document.addEventListener("keydown", (event) => {
    if (!run) return;
    if (event.key === "Escape") {
      event.preventDefault();
      endRun(false);
      return;
    }
    // Space on a focused button is that button's click: let the one that
    // means "next" through, and keep every other from firing twice.
    const typing = event.target.closest?.("input, textarea, select, [contenteditable]");
    if (event.code === "Space" && !typing) {
      event.preventDefault();
      nextBlock();
    }
  });

  // --- recording ----------------------------------------------------------- //

  let rec = null; // { block, button, recorder, stream, timer, cancelled }
  const takeUrls = new Map(); // block index -> object URL of its latest take

  function silence() {
    document.dispatchEvent(new CustomEvent("tc:silence"));
  }

  function pickMimeType() {
    return MIME_TYPES.find((type) => window.MediaRecorder.isTypeSupported(type)) ?? "";
  }

  function say(block, text) {
    const box = block.head.querySelector(".block-take");
    box.replaceChildren(cell("span", text, "heard failed"));
  }

  function release(r) {
    clearTimeout(r.timer);
    for (const track of r.stream?.getTracks() ?? []) track.stop();
    r.button.textContent = "🎙 Record yourself";
    r.button.setAttribute("aria-pressed", "false");
    r.button.classList.remove("recording");
  }

  function stopRecording() {
    if (!rec) return;
    const r = rec;
    rec = null;
    if (r.recorder && r.recorder.state !== "inactive") r.recorder.stop();
    else r.cancelled = true; // still waiting for the permission
    release(r);
  }

  async function record(block, button) {
    const same = rec?.block === block;
    stopRecording();
    if (same) return;
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      say(block, "This browser cannot record.");
      return;
    }
    silence();
    const r = { block, button, cancelled: false };
    rec = r;
    button.textContent = "⏺ Recording… click to stop";
    button.setAttribute("aria-pressed", "true");
    button.classList.add("recording");
    try {
      r.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
    } catch (error) {
      if (rec === r) rec = null;
      release(r);
      say(block, `No microphone: ${error.message}`);
      return;
    }
    if (r.cancelled) {
      release(r);
      return;
    }
    const mimeType = pickMimeType();
    const chunks = [];
    r.recorder = new MediaRecorder(r.stream, mimeType ? { mimeType } : undefined);
    r.recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    });
    r.recorder.addEventListener("stop", () => {
      const blob = new Blob(chunks, { type: r.recorder.mimeType || mimeType || "audio/webm" });
      if (!blob.size) {
        say(block, "Nothing was recorded.");
        return;
      }
      // Only the latest take is kept: the one before it is let go.
      const old = takeUrls.get(block.index);
      if (old) URL.revokeObjectURL(old);
      const url = URL.createObjectURL(blob);
      takeUrls.set(block.index, url);
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = url;
      audio.setAttribute("aria-label", `Your take of “${block.title}”`);
      block.head.querySelector(".block-take").replaceChildren(audio);
      button.textContent = "🎙 Record again";
    });
    r.recorder.start();
    r.timer = setTimeout(() => {
      if (rec === r) stopRecording();
    }, MAX_TAKE_MS);
  }

  // --- turning it on ------------------------------------------------------- //

  // Sticky measures from the scroller's padding edge, and the lesson's <main>
  // has top padding: at `top: 0` the text showed through above the bar. The
  // same correction lesson.js's dockChart makes at the bottom.
  const scroller = root.closest("main");
  if (scroller) {
    const above = parseFloat(getComputedStyle(scroller).paddingTop) || 0;
    root.style.top = `${-above}px`;
  }

  function applyOn() {
    root.hidden = !state.on;
    notes.hidden = !state.on;
    toggle.setAttribute("aria-pressed", String(state.on));
    reading.classList.toggle("rehearsing", state.on);
    if (!state.on) {
      endRun(false);
      stopWatch(false);
      stopRecording();
    }
    applyCues();
  }

  toggle.addEventListener("click", () => {
    state.on = !state.on;
    applyOn();
    save();
  });

  reading.addEventListener("click", (event) => {
    const button = event.target.closest("button.time-block, button.record-block");
    if (!button) return;
    const block = blocks.find((b) => b.head.contains(button));
    if (!block) return;
    if (button.matches(".time-block")) timeBlock(block, button);
    else record(block, button);
  });

  window.addEventListener("pagehide", () => {
    stopRecording();
    for (const url of takeUrls.values()) URL.revokeObjectURL(url);
  });

  refresh();
  drawRuns();
  applyOn();
})();

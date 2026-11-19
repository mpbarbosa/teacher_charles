"use strict";

// The speaker and the microphone beside each word on a lesson page.
//
// Only loaded when the server has a voice (the template leaves the buttons and
// this script out otherwise). One delegated listener, so the buttons need no
// wiring of their own.
//
// A word's audio is kept after the first click: the student presses the same
// word several times while practising it, and Kokoro would synthesise the same
// sound each time. The cache lives as long as the page, which is one lesson.
//
// ## What the microphone is, and what it is not
//
// It records the student saying the word and puts the take beside Charles's,
// so the two can be heard one after the other and compared by ear. It is not a
// pronunciation check: that is E10/E11 of docs/plano-fase-2.md, a phoneme
// recognizer and an evaluator that reports only errors with a curated rule,
// and neither exists yet.
//
// The take also goes through /api/stt, and what Whisper made of it is shown
// under the word. That is recognition and it is worded as recognition: Whisper
// picks the likeliest English word, so it can understand a mispronounced word
// and miss a well-said one, and its accuracy on a Brazilian accent has never
// been measured (plano 1.5 §2.3). No "correct", no score — and the reading is
// held against what Whisper hears from Charles's own audio of the same word,
// because on some words it misses even that (see `controlFor`).
//
// /api/stt records no turn, so a practice take never becomes a correction, a
// review or a level signal. The take itself never leaves the page: it lives in
// memory and dies with it.
//
// ## The reader
//
// The button beside "Full text" reads the whole text aloud, in a voice that is
// not Charles's (the page names it in `data-voice`). /api/tts takes one
// sentence per call -- 600 characters at most, the same ceiling the chat's
// voice lives under -- so the text goes out sentence by sentence, the next one
// synthesising while the current one plays. The sentence being spoken is lit,
// in the text and in its IPA line, so the eye can follow the voice. Nothing of
// it is cached: a reading is minutes of audio, and a second one is rare enough
// to pay for again.

(() => {
  const cache = new Map(); // word -> Promise<Blob>, Charles's audio
  const takes = new Map(); // word -> Blob, the student's latest take
  let current = null; // { audio, button, url, finish }
  // Bumped by every stop, so "Charles, then you" gives up on its second half
  // when anything else -- a playback, a recording -- started in the gap.
  let playSeq = 0;

  // The longest take kept running by itself: one word or a short phrase, and a
  // forgotten click should not record the room for a minute.
  const MAX_TAKE_MS = 5000;
  // Shorter than this is a double click, not a word.
  const MIN_TAKE_MS = 350;
  // Opus in webm (Chromium) or ogg (Firefox), the same list voice.js uses;
  // PyAV decodes both on the server.
  const MIME_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/ogg"];

  // --- playback ------------------------------------------------------------ //

  /** Silences everything: whatever is playing, and the reader with it. */
  function stop() {
    endReading();
    halt();
  }

  /** Silences what is playing, but leaves the reader's place alone. */
  function halt() {
    playSeq += 1;
    if (!current) return;
    const { audio, button, url, finish } = current;
    current = null;
    audio.pause();
    URL.revokeObjectURL(url);
    button.classList.remove("playing");
    finish(false);
  }

  /** Plays a blob, lighting `button`. Resolves true at the end, false if cut. */
  function play(blob, button) {
    halt();
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      current = { audio, button, url, finish: resolve };
      button.classList.add("playing");
      const ended = () => {
        if (current?.audio !== audio) return;
        current.finish = () => {};
        halt();
        resolve(true);
      };
      audio.addEventListener("ended", ended);
      audio.addEventListener("error", ended);
      audio.play().catch((error) => {
        if (current?.audio === audio) {
          current.finish = () => {};
          halt();
        }
        reject(error);
      });
    });
  }

  /** One sentence as audio. `voice` omitted is Charles's. */
  function speak(text, voice) {
    return fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(voice ? { text, voice } : { text }),
    }).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.blob();
    });
  }

  function fetchWord(word) {
    if (!cache.has(word)) {
      const pending = speak(word);
      // A failure is not cached: the next click tries again.
      pending.catch(() => cache.delete(word));
      cache.set(word, pending);
    }
    return cache.get(word);
  }

  /** Charles's audio for a word, with `button` showing the wait and the failure. */
  async function charles(word, button) {
    button.classList.add("loading");
    try {
      const blob = await fetchWord(word);
      button.classList.remove("failed");
      return blob;
    } catch (error) {
      button.classList.add("failed");
      button.title = `Charles lost his voice: ${error.message}`;
      return null;
    } finally {
      button.classList.remove("loading");
    }
  }

  function blocked(button, error) {
    button.classList.add("failed");
    button.title = `The browser blocked the sound: ${error.message}`;
  }

  async function say(button) {
    const wasThis = current?.button === button;
    stop();
    // A second click on the word that is playing is "stop", not "again".
    if (wasThis) return;
    button.title = "Hear it";
    const blob = await charles(button.dataset.say, button);
    if (!blob) return;
    play(blob, button).catch((error) => blocked(button, error));
  }

  async function playMine(button) {
    const wasThis = current?.button === button;
    stop();
    if (wasThis) return;
    const take = takes.get(button.dataset.word);
    if (take) play(take, button).catch((error) => blocked(button, error));
  }

  async function playBoth(button) {
    const wasThis = current?.button === button;
    stop();
    if (wasThis) return;
    const word = button.dataset.word;
    const take = takes.get(word);
    if (!take) return;
    const blob = await charles(word, button);
    if (!blob) return;
    try {
      if (!(await play(blob, button))) return;
      const seq = playSeq;
      // A breath between the two, or they run together into one word.
      await new Promise((resolve) => setTimeout(resolve, 350));
      if (seq !== playSeq) return;
      await play(take, button);
    } catch (error) {
      blocked(button, error);
    }
  }

  // --- recording ----------------------------------------------------------- //

  // { button, word, recorder, stream, timer, startedAt, cancelled }
  let rec = null;

  function pickMimeType() {
    for (const type of MIME_TYPES) {
      if (window.MediaRecorder.isTypeSupported(type)) return type;
    }
    return "";
  }

  /** The row under the word where the take and Whisper's reading go. */
  function takeRow(button) {
    const cell = button.closest("td") ?? button.parentElement;
    let row = cell.querySelector(".take");
    if (!row) {
      row = document.createElement("div");
      row.className = "take";
      row.setAttribute("role", "status");
      cell.append(row);
    }
    return row;
  }

  function showStatus(row, text, kind) {
    row.replaceChildren();
    const span = document.createElement("span");
    span.className = `heard${kind ? ` ${kind}` : ""}`;
    span.textContent = text;
    row.append(span);
  }

  function release(r) {
    clearTimeout(r.timer);
    for (const track of r.stream?.getTracks() ?? []) track.stop();
    r.button.classList.remove("recording");
    r.button.title = "Record yourself";
  }

  function finishRecording() {
    if (!rec) return;
    const r = rec;
    rec = null;
    if (r.recorder) {
      r.recorder.stop(); // the "stop" event delivers the take
    } else {
      r.cancelled = true; // still waiting for the microphone permission
    }
    release(r);
  }

  async function record(button) {
    if (rec) {
      const same = rec.button === button;
      finishRecording();
      if (same) return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      showStatus(takeRow(button), "This browser cannot record.", "failed");
      return;
    }
    // Charles's voice would go into the microphone with the student's.
    stop();

    const r = { button, word: button.dataset.word, cancelled: false };
    rec = r;
    button.classList.add("recording");
    button.title = "Stop";
    try {
      r.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
    } catch (error) {
      if (rec === r) rec = null;
      release(r);
      showStatus(takeRow(button), `No microphone: ${error.message}`, "failed");
      return;
    }
    // The permission takes as long as the student takes to click "allow", and
    // the button may have been clicked again meanwhile.
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
      const elapsed = Date.now() - r.startedAt;
      const blob = new Blob(chunks, { type: r.recorder.mimeType || mimeType || "audio/webm" });
      received(r.button, r.word, blob, elapsed);
    });
    r.recorder.start();
    r.startedAt = Date.now();
    r.timer = setTimeout(() => {
      if (rec === r) finishRecording();
    }, MAX_TAKE_MS);
  }

  function makeButton(className, label, word, text) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = className;
    b.dataset.word = word;
    b.setAttribute("aria-label", label);
    b.title = label;
    b.textContent = text;
    return b;
  }

  async function received(recordButton, word, blob, elapsed) {
    const row = takeRow(recordButton);
    if (elapsed < MIN_TAKE_MS || blob.size === 0) {
      showStatus(row, "Too short — click, say the word, then click again.", "failed");
      return;
    }
    takes.set(word, blob);

    const heard = document.createElement("span");
    heard.className = "heard";
    heard.textContent = "Whisper is listening…";
    heard.title =
      "What a speech recognizer made of your take. Not a pronunciation score: " +
      "Whisper picks the likeliest word, so it can understand a mispronounced " +
      "word and miss a well-said one.";
    row.replaceChildren(
      makeButton("mine", `Play your take of “${word}”`, word, "▶ you"),
      makeButton("both", `Play Charles, then your take of “${word}”`, word, "▶ Charles, then you"),
      heard
    );

    // The control runs beside the take, not after it: it is cached per word,
    // so from the second take on it costs nothing.
    const control = controlFor(word);
    let text;
    try {
      text = await transcribe(blob);
    } catch (error) {
      // A newer take of the same word has its own row content by now.
      if (takes.get(word) !== blob) return;
      heard.classList.add("failed");
      heard.textContent = `Whisper could not listen: ${error.message}`;
      return;
    }
    const fromCharles = await control;
    if (takes.get(word) !== blob) return;

    const verdict = reading(word, text, fromCharles);
    if (verdict.kind) heard.classList.add(verdict.kind);
    heard.textContent = verdict.text;
  }

  /** What one take sounds like to Whisper, as text. */
  async function transcribe(blob) {
    const response = await fetch("/api/stt", { method: "POST", body: blob });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail ?? `HTTP ${response.status}`);
    return (payload.text ?? "").trim();
  }

  // ## Why Charles's own audio goes through Whisper too
  //
  // Measured on the first lesson (16 words), Whisper base.en understood
  // Kokoro's audio of 11: "cleave" came back "Cleaver", "abyss" "A Biz",
  // "truth" "Truth in". That audio is the reference the student imitates, so on
  // those five a perfect imitation would read "Whisper heard something else" --
  // a line that says "you got it wrong" to a student who got it right.
  //
  // So every word gets a control: Charles's audio, transcribed once per page.
  // When Whisper misses the control, the student's line says so, and what it
  // heard from the student is shown as information rather than a miss. The
  // control does not make Whisper a judge either way; it only stops it from
  // being a judge of the words it cannot hear at all.
  const controls = new Map(); // word -> Promise<string | null>

  function controlFor(word) {
    if (!controls.has(word)) {
      const pending = fetchWord(word)
        .then(transcribe)
        // No control is not a failed take: the line falls back to the plain
        // reading, and the next take tries again.
        .catch(() => {
          controls.delete(word);
          return null;
        });
      controls.set(word, pending);
    }
    return controls.get(word);
  }

  /** The line under the take: what Whisper heard, read against the control. */
  function reading(word, text, fromCharles) {
    const said = tidy(text);
    if (!said) return { text: "Whisper heard nothing — try a little louder or closer." };
    const understood = normalise(text) === normalise(word);
    if (fromCharles !== null && normalise(fromCharles) !== normalise(word)) {
      return {
        kind: "uncheckable",
        text:
          `Whisper heard “${said}” — but it hears Charles as “${tidy(fromCharles)}” ` +
          "too, so on this word only your ears can compare.",
      };
    }
    if (understood) return { kind: "understood", text: `Whisper understood “${said}”.` };
    return { text: `Whisper heard “${said}”.` };
  }

  /** Whisper ends a lone word with a period; inside quotes it reads as noise. */
  function tidy(text) {
    return (text ?? "").trim().replace(/[\s.!?,;:…]+$/u, "");
  }

  /** Case, punctuation and spacing are not what the line is about. */
  function normalise(text) {
    return text
      .toLowerCase()
      .replace(/[’‘]/g, "'")
      .replace(/[^a-z' ]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  // --- the reader ---------------------------------------------------------- //

  // Under the route's 600 with room to spare: a sentence longer than this is
  // cut at a word, and its parts light the same sentence one after another.
  const READ_MAX_CHARS = 400;

  let reader = null; // { button, label }, while the reader has the floor

  /** A sentence longer than the ceiling, cut at the last space that fits. */
  function wrap(text) {
    const out = [];
    let rest = text.trim();
    while (rest.length > READ_MAX_CHARS) {
      const cut = rest.lastIndexOf(" ", READ_MAX_CHARS);
      const at = cut > 0 ? cut : READ_MAX_CHARS;
      out.push(rest.slice(0, at).trim());
      rest = rest.slice(at).trim();
    }
    if (rest) out.push(rest);
    return out;
  }

  /**
   * Lights `sentence` and its IPA, and only them. The two share a `data-s`
   * (the template cuts both lines into the same sentences), so the IPA is
   * lit whether its line is showing or not -- shown later, it is already on
   * the right place.
   */
  function mark(sentence) {
    const lit = sentence
      ? document.querySelectorAll(`.full-text [data-s="${sentence.dataset.s}"]`)
      : [];
    for (const el of document.querySelectorAll(".full-text .now-reading")) {
      if (![...lit].includes(el)) el.classList.remove("now-reading");
    }
    for (const el of lit) el.classList.add("now-reading");
  }

  function endReading() {
    if (!reader) return;
    const { button, label } = reader;
    reader = null;
    button.textContent = label;
    button.classList.remove("loading");
    button.setAttribute("aria-pressed", "false");
    mark(null);
  }

  async function readAloud(button) {
    const wasReading = reader?.button === button;
    stop();
    // A second click while it reads is "stop", not "from the top".
    if (wasReading) return;

    // One piece per sentence the server cut, so what is heard, what is lit
    // in the text and what is lit in the IPA are the same sentence.
    const queue = [...document.querySelectorAll(".full-text .reading .sentence")].flatMap(
      (sentence) => wrap(sentence.textContent).map((text) => ({ sentence, text }))
    );
    if (!queue.length) return;

    const voice = button.dataset.voice;
    const run = { button, label: button.textContent };
    reader = run;
    button.classList.remove("failed");
    button.title = "Stop reading";
    button.setAttribute("aria-pressed", "true");
    button.textContent = "■ Stop";

    // The next piece synthesises while this one plays. A failure waits to be
    // awaited, so a piece the student never reaches never reports one.
    const ahead = (i) => {
      if (i >= queue.length) return null;
      const pending = speak(queue[i].text, voice);
      pending.catch(() => {});
      return pending;
    };
    let next = ahead(0);
    for (let i = 0; i < queue.length; i += 1) {
      let blob;
      button.classList.add("loading");
      try {
        blob = await next;
      } catch (error) {
        if (reader !== run) return;
        endReading();
        button.classList.add("failed");
        button.title = `The reader lost her voice: ${error.message}`;
        return;
      } finally {
        button.classList.remove("loading");
      }
      if (reader !== run) return;
      next = ahead(i + 1);
      mark(queue[i].sentence);
      queue[i].sentence.scrollIntoView({ block: "nearest", behavior: "smooth" });
      try {
        if (!(await play(blob, button))) return; // cut: stop() already tidied up
      } catch (error) {
        if (reader === run) endReading();
        blocked(button, error);
        return;
      }
      if (reader !== run) return;
    }
    if (reader === run) {
      endReading();
      button.title = "Listen to the full text";
    }
  }

  // --- wiring -------------------------------------------------------------- //

  document.addEventListener("click", (event) => {
    const target = event.target.closest(
      "button.say, button.record, button.mine, button.both, button.read-aloud"
    );
    if (!target) return;
    if (target.matches(".read-aloud")) readAloud(target);
    else if (target.matches(".say")) say(target);
    else if (target.matches(".record")) record(target);
    else if (target.matches(".mine")) playMine(target);
    else playBoth(target);
  });
})();

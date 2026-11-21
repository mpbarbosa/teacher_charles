"use strict";

// The phonetics page: a sentence goes in, its IPA comes out.
//
// The sentence is POSTed to /api/phonetics (never put in the URL: it is what
// the student typed). The answer is one entry per line of the box, each a list
// of sentences with the whole sentence's IPA and its words paired with their
// own IPA. Words espeak runs together ("on the" -> ɔnðə) arrive as one pair.
// Everything is drawn with textContent, so nothing typed becomes markup.
//
// With a voice on the server (`data-voice` on the result), each sentence gets
// a Listen button, and "Listen to all" reads every sentence in order when
// there is more than one. /api/tts takes one sentence per call -- that is its
// design, and its 600-character ceiling -- so the next sentence is asked for
// while the current one plays, and the block being read is lit. The voice is
// Charles's Kokoro, which speaks through the same en-us espeak that wrote the
// IPA, so what plays is the transcription on screen read aloud.
//
// While a sentence plays, the word being said is lit together with its IPA
// (the pair holds both), the way the lesson reader lights its words. Where the
// words fall in the audio is measured by /api/tts/{id}/words, a Whisper pass
// asked for as soon as the audio is in; until it answers, the light follows
// word-starts.js's estimate from the words' letters, and the measured starts
// take over mid-word if need be. Words espeak joined into one pair light the
// pair while either is said.
//
// How fast it all goes is the speed bar's (speech-speed.js), the same bar and
// the same remembered setting as on a lesson. The next sentence is fetched
// while this one plays, so a new voice speed is heard from the sentence after
// next; a new playback rate at once.

(() => {
  const form = document.getElementById("phonetics-form");
  const input = document.getElementById("phonetics-text");
  const result = document.getElementById("phonetics-result");
  if (!form || !input || !result) return;
  const submit = form.querySelector("button[type=submit]");
  const hasVoice = result.dataset.voice === "on";
  // /api/tts's ceiling (SpeakRequest): a longer sentence is refused there.
  const TTS_MAX_CHARS = 600;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function sentenceBlock(sentence) {
    const block = el("div", "spelled");

    if (sentence.pairs.length) {
      const words = el("p", "pairs");
      // Word k of the sentence (split on whitespace, as /api/tts/{id}/words
      // counts them) -> the cell that shows it. A pair joins its words with
      // single spaces, so its word count is its text split on them.
      const cells = [];
      for (const pair of sentence.pairs) {
        const cell = el("span", "pair");
        cell.append(el("span", "word", pair.text), el("span", "word-ipa", pair.ipa || "\u00a0"));
        words.append(cell);
        for (const _ of pair.text.split(" ")) cells.push(cell);
      }
      wordCells.set(block, cells);
      block.append(words);
    } else {
      block.append(el("p", "said", sentence.text));
    }

    const line = el("p", "ipa whole", sentence.ipa ? `/${sentence.ipa}/` : "(no sounds to transcribe)");
    line.lang = "und-fonipa";
    block.append(line);

    const copy = el("button", "copy", "Copy IPA");
    copy.type = "button";
    copy.disabled = !sentence.ipa;
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(sentence.ipa);
        copy.textContent = "Copied";
      } catch {
        copy.textContent = "Copy failed";
      }
      setTimeout(() => { copy.textContent = "Copy IPA"; }, 1500);
    });
    block.append(copy);

    if (hasVoice) {
      const listen = el("button", "listen", LISTEN);
      listen.type = "button";
      listen.setAttribute("aria-pressed", "false");
      if (!speakable(sentence)) {
        listen.disabled = true;
        listen.title = sentence.ipa ? "Too long to read aloud in one go" : "Nothing to say";
      }
      listen.addEventListener("click", () => toggle(listen, [{ text: sentence.text, block }], LISTEN));
      block.append(listen);
    }
    return block;
  }

  // --- listening ------------------------------------------------------------ //

  // A sentence's block -> the cell of each of its words; absent when the
  // sentence came back without pairs (a run-on), which then lights whole.
  const wordCells = new WeakMap();

  const LISTEN = "\u{1F50A} Listen";
  const LISTEN_ALL = "\u{1F50A} Listen to all";
  const STOP = "\u25A0 Stop";

  function speakable(sentence) {
    return Boolean(sentence.ipa) && sentence.text.length <= TTS_MAX_CHARS;
  }

  /**
   * One sentence in Charles's voice: `{ blob, starts }`, where `starts` is a
   * promise of when each word begins, in seconds. Rejects with the server's
   * reason. The timings are a second request, made the moment the audio is
   * in, because they cost a Whisper pass the sound should not wait for; they
   * resolve to [] when the server could not time the sentence, and are not
   * asked for at all when there are no word cells to light (`timed` false).
   */
  async function speak(text, timed) {
    // The speed bar's voice speed (speech-speed.js); absent, the server's.
    const speed = window.TCSpeechSpeed?.voiceSpeed();
    const response = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(timed ? { text, speed, timings: true } : { text, speed }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(typeof body.detail === "string" ? body.detail : `the server answered ${response.status}`);
    }
    const id = response.headers.get("X-Speech-Id");
    const starts = id
      ? fetch(`/api/tts/${encodeURIComponent(id)}/words`)
          .then((r) => (r.ok ? r.json() : { starts: [] }))
          .then((body) => (Array.isArray(body.starts) ? body.starts : []))
          .catch(() => [])
      : Promise.resolve([]);
    return { blob: await response.blob(), starts };
  }

  /**
   * Keeps the lit cell on the word being said while `audio` plays, and
   * returns the function that stops following and clears the light.
   *
   * Driven by animation frames, which track the voice closely, and by the
   * audio's `timeupdate`, a few times a second: frames stop in a hidden tab,
   * and the light must not freeze when the student looks away.
   */
  function follow(audio, cells, timings, text) {
    let starts = [];
    let measured = false;
    let lit = null;
    let live = true;
    const light = (cell) => {
      if (cell === lit) return;
      if (lit) lit.classList.remove("now-word");
      if (cell) cell.classList.add("now-word");
      lit = cell;
    };
    const update = () => {
      if (!live || !starts.length) return;
      const k = window.TCWordStarts.at(starts, audio.currentTime);
      light(k < 0 ? null : cells[Math.min(k, cells.length - 1)]);
    };
    const estimate = () => {
      if (measured || !live) return;
      starts = window.TCWordStarts.estimate(text.trim().split(/\s+/), audio.duration);
      update();
    };
    const frame = () => {
      if (!live) return;
      update();
      requestAnimationFrame(frame);
    };
    audio.addEventListener("timeupdate", update);
    // A WAV blob knows its length almost at once; it may already.
    if (Number.isFinite(audio.duration)) estimate();
    else audio.addEventListener("loadedmetadata", estimate, { once: true });
    timings.then((resolved) => {
      if (!live || !resolved.length) return;
      measured = true;
      starts = resolved;
      update();
    });
    requestAnimationFrame(frame);
    return () => {
      live = false;
      audio.removeEventListener("timeupdate", update);
      light(null);
    };
  }

  // The one reading in progress: starting another, or showing a new result,
  // stops it first, so two voices never talk over each other.
  let reading = null;

  function stopReading() {
    if (reading) reading.stop();
  }

  function toggle(button, parts, label) {
    const mine = reading && reading.button === button;
    stopReading();
    if (!mine) read(button, parts, label);
  }

  async function read(button, parts, label) {
    let stopped = false;
    let audio = null;
    let wake = null;
    const me = {
      button,
      stop() {
        stopped = true;
        if (audio) audio.pause();
        if (wake) wake();
      },
    };
    reading = me;
    button.classList.remove("failed");
    button.removeAttribute("title");
    button.textContent = STOP;
    button.setAttribute("aria-pressed", "true");

    // The next sentence is synthesised while this one plays: Kokoro takes
    // about as long to make a sentence as to say it, and waiting for it after
    // the sound ends would put a gap between every two sentences.
    const pending = parts.map(() => null);
    const fetchPart = (i) => {
      if (i < parts.length && !pending[i]) {
        pending[i] = speak(parts[i].text, wordCells.has(parts[i].block));
        pending[i].catch(() => {}); // surfaced when it is awaited, not here
      }
      return pending[i];
    };

    try {
      for (let i = 0; i < parts.length && !stopped; i += 1) {
        const { blob, starts } = await fetchPart(i);
        if (stopped) break;
        fetchPart(i + 1);
        const url = URL.createObjectURL(blob);
        const cells = wordCells.get(parts[i].block);
        let unfollow = null;
        parts[i].block.classList.add("now-reading");
        try {
          audio = new Audio(url);
          window.TCSpeechSpeed?.applyRate(audio);
          if (cells) unfollow = follow(audio, cells, starts, parts[i].text);
          await new Promise((resolve, reject) => {
            wake = resolve;
            audio.addEventListener("ended", resolve);
            audio.addEventListener("error", () => reject(new Error("the audio would not play")));
            audio.play().catch(reject);
          });
        } finally {
          if (unfollow) unfollow();
          parts[i].block.classList.remove("now-reading");
          URL.revokeObjectURL(url);
          audio = null;
          wake = null;
        }
      }
    } catch (error) {
      if (!stopped) {
        button.title = `Charles lost his voice: ${error.message}`;
        button.classList.add("failed");
      }
    } finally {
      if (reading === me) reading = null;
      button.textContent = label;
      button.setAttribute("aria-pressed", "false");
    }
  }

  /** "Listen to all", above the sentences, when there is more than one to say. */
  function listenToAll(blocks) {
    const parts = blocks.filter((b) => speakable(b.sentence)).map((b) => ({ text: b.sentence.text, block: b.block }));
    if (parts.length < 2) return null;
    const bar = el("div", "listen-all");
    const button = el("button", "listen", LISTEN_ALL);
    button.type = "button";
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => toggle(button, parts, LISTEN_ALL));
    bar.append(button);
    return bar;
  }

  function show(body) {
    stopReading();
    result.replaceChildren();
    const blocks = [];
    for (const line of body.lines) {
      for (const sentence of line) blocks.push({ sentence, block: sentenceBlock(sentence) });
    }
    const all = hasVoice ? listenToAll(blocks) : null;
    if (all) result.append(all);
    for (const { block } of blocks) result.append(block);
    if (!blocks.length) result.append(el("p", "lead empty", "Nothing to transcribe."));
    result.hidden = false;
  }

  function fail(message) {
    stopReading();
    result.replaceChildren(el("p", "error", message));
    result.hidden = false;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    submit.disabled = true;
    submit.textContent = "Working…";
    try {
      const response = await fetch("/api/phonetics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof body.detail === "string" ? body.detail : `the server answered ${response.status}`;
        fail(`Could not get the IPA: ${detail}.`);
        return;
      }
      show(body);
    } catch {
      fail("Could not reach Charles. Is the server still running?");
    } finally {
      submit.disabled = false;
      submit.textContent = "Show phonetics";
    }
  });

  // Enter shows the IPA; Shift+Enter breaks the line — the chat box's rule.
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
})();

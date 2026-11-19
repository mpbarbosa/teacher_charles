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
      for (const pair of sentence.pairs) {
        const cell = el("span", "pair");
        cell.append(el("span", "word", pair.text), el("span", "word-ipa", pair.ipa || "\u00a0"));
        words.append(cell);
      }
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

  const LISTEN = "\u{1F50A} Listen";
  const LISTEN_ALL = "\u{1F50A} Listen to all";
  const STOP = "\u25A0 Stop";

  function speakable(sentence) {
    return Boolean(sentence.ipa) && sentence.text.length <= TTS_MAX_CHARS;
  }

  /** One sentence as WAV, in Charles's voice. Rejects with the server's reason. */
  async function speak(text) {
    const response = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(typeof body.detail === "string" ? body.detail : `the server answered ${response.status}`);
    }
    return response.blob();
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
        pending[i] = speak(parts[i].text);
        pending[i].catch(() => {}); // surfaced when it is awaited, not here
      }
      return pending[i];
    };

    try {
      for (let i = 0; i < parts.length && !stopped; i += 1) {
        const blob = await fetchPart(i);
        if (stopped) break;
        fetchPart(i + 1);
        const url = URL.createObjectURL(blob);
        parts[i].block.classList.add("now-reading");
        try {
          audio = new Audio(url);
          await new Promise((resolve, reject) => {
            wake = resolve;
            audio.addEventListener("ended", resolve);
            audio.addEventListener("error", () => reject(new Error("the audio would not play")));
            audio.play().catch(reject);
          });
        } finally {
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

"use strict";

// The speaker beside each word on a lesson page.
//
// Only loaded when the server has a voice (the template leaves both the
// buttons and this script out otherwise). One delegated listener, so the
// buttons need no wiring of their own.
//
// A word's audio is kept after the first click: the student presses the same
// word several times while practising it, and Kokoro would synthesise the same
// sound each time. The cache lives as long as the page, which is one lesson.

(() => {
  const cache = new Map(); // word -> Promise<Blob>
  let current = null; // { audio, button, url }

  function stop() {
    if (!current) return;
    current.audio.pause();
    URL.revokeObjectURL(current.url);
    current.button.classList.remove("playing");
    current = null;
  }

  function fetchWord(word) {
    if (!cache.has(word)) {
      const pending = fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: word }),
      }).then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.blob();
      });
      // A failure is not cached: the next click tries again.
      pending.catch(() => cache.delete(word));
      cache.set(word, pending);
    }
    return cache.get(word);
  }

  async function say(button) {
    const word = button.dataset.say;
    const wasThis = current?.button === button;
    stop();
    // A second click on the word that is playing is "stop", not "again".
    if (wasThis) return;

    button.classList.add("loading");
    button.title = "Hear it";
    let blob;
    try {
      blob = await fetchWord(word);
    } catch (error) {
      button.classList.remove("loading");
      button.classList.add("failed");
      button.title = `Charles lost his voice: ${error.message}`;
      return;
    }
    button.classList.remove("loading", "failed");

    stop();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    current = { audio, button, url };
    button.classList.add("playing");
    const done = () => {
      if (current?.audio === audio) stop();
    };
    audio.addEventListener("ended", done);
    audio.addEventListener("error", done);
    audio.play().catch((error) => {
      done();
      button.classList.add("failed");
      button.title = `The browser blocked the sound: ${error.message}`;
    });
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest("button.say");
    if (button) say(button);
  });
})();

"use strict";

// The speed bar (templates/_speech_speed.html), shared by the lesson page and
// the phonetics page, and exposed to their scripts as window.TCSpeechSpeed.
//
// Two knobs, because they are two different things. `voiceSpeed()` is
// Kokoro's own speaking rate, sent with each /api/tts request: the voice
// really says the words slower, so it stays natural at the slow end, but only
// sounds synthesised after the change hear it -- and a page that caches audio
// must key the cache on it. `applyRate(audio)` is the browser's playbackRate
// on a finished sound, with the pitch kept: it takes effect at once, on what
// is already playing too, at the cost of a stretched sound when slow. They
// multiply. A page's word highlighting reads the audio's own clock
// (`currentTime`), so it stays in step under both.
//
// Both are remembered in this browser, under the same keys on every page, so
// a speed chosen on a lesson is the speed of the phonetics page too.

(() => {
  const SPEED_KEY = "charles.voiceSpeed";
  const RATE_KEY = "charles.playbackRate";
  const bar = document.querySelector(".speech-speed");
  const speedInput = bar?.querySelector("input.voice-speed");
  const rateInput = bar?.querySelector("input.playback-rate");
  const defaultSpeed = Number(bar?.dataset.defaultSpeed) || 1;
  // The sounds playing now: a Playback change reaches them mid-sentence.
  const playing = new Set();

  function stored(key) {
    try {
      const value = Number(localStorage.getItem(key));
      return Number.isFinite(value) && value > 0 ? value : null;
    } catch {
      return null;
    }
  }

  function remember(key, value) {
    try {
      localStorage.setItem(key, String(value));
    } catch {
      /* no storage: the knob still works, for this page */
    }
  }

  /** Kokoro's rate for the next request. Two decimals, so it keys a cache. */
  function voiceSpeed() {
    return speedInput ? Number(Number(speedInput.value).toFixed(2)) : defaultSpeed;
  }

  function playbackRate() {
    return rateInput ? Number(rateInput.value) : 1;
  }

  function setRate(audio) {
    const rate = playbackRate();
    audio.defaultPlaybackRate = rate;
    audio.playbackRate = rate;
    audio.preservesPitch = true;
  }

  /**
   * The browser's rate on `audio`, kept up to date while it plays. Call it
   * before `play()`. Stopping on these pages is a pause, never a resume, so a
   * pause ends the tracking as an end does.
   */
  function applyRate(audio) {
    setRate(audio);
    playing.add(audio);
    const done = () => playing.delete(audio);
    for (const type of ["ended", "pause", "error"]) audio.addEventListener(type, done, { once: true });
  }

  function show() {
    bar.querySelector(".voice-speed-value").textContent = `${voiceSpeed().toFixed(2)}×`;
    bar.querySelector(".playback-rate-value").textContent = `${playbackRate().toFixed(2)}×`;
  }

  function rateChanged() {
    remember(RATE_KEY, playbackRate());
    show();
    for (const audio of playing) setRate(audio);
  }

  if (bar) {
    // A stored value outside the slider is clamped by the browser itself.
    const speed = stored(SPEED_KEY);
    if (speed !== null) speedInput.value = String(speed);
    const rate = stored(RATE_KEY);
    if (rate !== null) rateInput.value = String(rate);
    show();

    speedInput.addEventListener("input", () => {
      remember(SPEED_KEY, voiceSpeed());
      show();
    });
    rateInput.addEventListener("input", rateChanged);
    bar.querySelector(".speed-reset").addEventListener("click", () => {
      speedInput.value = String(defaultSpeed);
      rateInput.value = "1";
      remember(SPEED_KEY, voiceSpeed());
      rateChanged();
    });
  }

  window.TCSpeechSpeed = { voiceSpeed, applyRate };
})();

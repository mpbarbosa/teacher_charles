"use strict";

// Follow me: lights the word of the full text the student is saying, while
// they read it aloud in the rehearsal.
//
// The browser cannot tell which word is being said, and nothing here guesses
// from the clock: the page listens. Every half second or so it sends the last
// few seconds of the microphone to /api/follow with the words around the
// light; the server's timing Whisper hears them, voice/follow.py finds where
// in those words the hearing ends, and the page lights that word. The light
// is about a second behind the voice -- the length of a Whisper pass -- and
// it moves only on something heard that is surely in the text, so a pause, a
// cough or a sentence off the script leaves it where it was.
//
// ## What goes where
//
// The audio goes to this app's own server and nowhere else, as 16 kHz PCM in
// the request body, and the server drops it with the request: it records
// nothing and sends back no transcript, only an index (app.py says why). A
// stretch that is quiet is not sent at all. The take that "Record yourself"
// keeps is a separate thing and still never leaves the page (rehearsal.js).
//
// ## Why the audio is captured raw
//
// A MediaRecorder slice after the first has no container header, so the
// server could not decode the last few seconds alone. The page keeps the raw
// samples from a ScriptProcessorNode instead, and downsamples them to the
// 16 kHz Whisper hears at. ScriptProcessorNode is deprecated in favour of
// AudioWorklet, but it runs in every browser, needs no second file, and for a
// tap that only copies samples the difference does not show.
//
// ## Lost and found
//
// While the light is lost -- at the start, or after a few windows in a row
// with sound and no place -- the request carries the whole text, so the
// student can start anywhere. Once found, it carries a window around the
// light: a few words behind, for a sentence said again, and forty ahead, more
// than anyone reads in a window. The short window is also what keeps a phrase
// the text says twice from being placed at its other copy.

(() => {
  const root = document.getElementById("rehearsal");
  const reading = document.getElementById("full-text-reading");
  const button = root?.querySelector("button.follow");
  const status = root?.querySelector(".follow-state");
  if (!root || !reading || !button || !status) return;

  // The text's words in reading order. The IPA line has `.w` too, but never
  // inside a `.sentence`.
  const words = [...reading.querySelectorAll(".sentence .w")];
  const texts = words.map((w) => w.textContent);
  if (!words.length) return;

  // The rate /api/follow takes, Whisper's own.
  const RATE = 16000;
  // How much of the microphone each request carries, in seconds: enough for
  // a few words to anchor on, little enough that Whisper is quick over it.
  const WINDOW_S = 3.5;
  // The pause after an answer before the next request goes.
  const EVERY_MS = 400;
  // The words sent around the light once it is found.
  const BEHIND = 8;
  const AHEAD = 40;
  // Windows in a row with sound and no place before the light counts as lost.
  const LOST_AFTER = 4;
  // The last second quieter than this is nobody reading: nothing is sent.
  const QUIET_RMS = 0.01;
  const QUIET_S = 1;

  let f = null; // { stream, ctx, chunks, length, keep, timer, pos, misses, lit, stopped }

  function show(text) {
    status.textContent = text;
  }

  function light(i) {
    if (f.lit === i) return;
    if (f.lit != null) words[f.lit].classList.remove("heard-word");
    f.lit = i;
    const el = words[i];
    el.classList.add("heard-word");
    // Scrolled to only once it leaves the view, so the light never fights a
    // student who is scrolling to look at something else. The sticky bar is
    // the top of the view.
    const box = el.getBoundingClientRect();
    if (box.top < root.getBoundingClientRect().bottom || box.bottom > window.innerHeight) {
      el.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }

  /** The kept samples as one array, the newest last. */
  function window_(me) {
    const out = new Float32Array(me.length);
    let at = 0;
    for (const chunk of me.chunks) {
      out.set(chunk, at);
      at += chunk.length;
    }
    return out;
  }

  function loud(samples, rate) {
    const n = Math.min(samples.length, Math.round(QUIET_S * rate));
    let sum = 0;
    for (let k = samples.length - n; k < samples.length; k++) sum += samples[k] * samples[k];
    return n > 0 && Math.sqrt(sum / n) >= QUIET_RMS;
  }

  /** To 16 kHz 16-bit PCM, each output sample the mean of the ones it covers. */
  function downsample(samples, rate) {
    const ratio = rate / RATE;
    const out = new Int16Array(Math.floor(samples.length / ratio));
    for (let i = 0; i < out.length; i++) {
      const from = Math.floor(i * ratio);
      const to = Math.max(from + 1, Math.floor((i + 1) * ratio));
      let sum = 0;
      for (let k = from; k < to; k++) sum += samples[k];
      const v = Math.max(-1, Math.min(1, sum / (to - from)));
      out[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
    }
    return out;
  }

  function toBase64(pcm) {
    const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    let binary = "";
    for (let k = 0; k < bytes.length; k += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(k, k + 0x8000));
    }
    return btoa(binary);
  }

  async function tick() {
    const me = f;
    if (!me) return;
    const rate = me.ctx.sampleRate;
    const samples = window_(me);
    if (samples.length >= rate && loud(samples, rate)) {
      const from = me.pos == null ? 0 : Math.max(0, me.pos - BEHIND);
      const to = me.pos == null ? texts.length : Math.min(texts.length, me.pos + AHEAD);
      try {
        const response = await fetch("/api/follow", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            audio: toBase64(downsample(samples, rate)),
            words: texts.slice(from, to),
          }),
        });
        if (me !== f) return;
        const body = await response.json().catch(() => ({}));
        if (!response.ok) {
          show(`Cannot follow: ${body.detail ?? response.status}`);
        } else if (body.at != null) {
          me.pos = from + body.at;
          me.misses = 0;
          light(me.pos);
          show("Following you.");
        } else if (me.pos != null && ++me.misses >= LOST_AFTER) {
          me.pos = null;
          show("Lost you: keep reading, and the light will find you.");
        }
      } catch (error) {
        if (me !== f) return;
        show(`Cannot follow: ${error.message}`);
      }
    }
    if (me === f) me.timer = setTimeout(tick, EVERY_MS);
  }

  function release(me) {
    clearTimeout(me.timer);
    for (const track of me.stream?.getTracks() ?? []) track.stop();
    me.ctx?.close().catch(() => {});
    if (me.lit != null) words[me.lit].classList.remove("heard-word");
  }

  function stop() {
    if (!f) return;
    const me = f;
    f = null;
    me.stopped = true; // a start still waiting for the permission ends there
    release(me);
    button.textContent = "👂 Follow me";
    button.setAttribute("aria-pressed", "false");
    show("");
  }

  async function start() {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) {
      show("This browser cannot listen.");
      return;
    }
    // The reader's voice in the loudspeaker would be followed instead.
    document.dispatchEvent(new CustomEvent("tc:silence"));
    const me = { chunks: [], length: 0, keep: 0, timer: 0, pos: null, misses: 0, lit: null, stopped: false };
    f = me;
    button.textContent = "■ Stop following";
    button.setAttribute("aria-pressed", "true");
    show("Opening the microphone…");
    try {
      me.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
      });
    } catch (error) {
      if (me === f) {
        stop();
        show(`No microphone: ${error.message}`);
      }
      return;
    }
    if (me.stopped) {
      release(me);
      return;
    }
    // At the device's own rate: Firefox refuses to connect a microphone to a
    // context of another rate, so the 16 kHz is made in downsample().
    const ctx = new AudioContext();
    me.ctx = ctx;
    await ctx.resume().catch(() => {});
    me.keep = Math.ceil(WINDOW_S * ctx.sampleRate);
    const source = ctx.createMediaStreamSource(me.stream);
    const tap = ctx.createScriptProcessor(4096, 1, 1);
    // A ScriptProcessorNode runs only when connected to the output; through
    // a gain of zero, so the student never hears themselves.
    const mute = ctx.createGain();
    mute.gain.value = 0;
    tap.onaudioprocess = (event) => {
      const data = event.inputBuffer.getChannelData(0);
      me.chunks.push(new Float32Array(data));
      me.length += data.length;
      while (me.chunks.length > 1 && me.length - me.chunks[0].length >= me.keep) {
        me.length -= me.chunks.shift().length;
      }
    };
    source.connect(tap);
    tap.connect(mute);
    mute.connect(ctx.destination);
    show("Listening: start reading anywhere in the text.");
    me.timer = setTimeout(tick, EVERY_MS);
  }

  button.addEventListener("click", () => (f ? stop() : start()));
  document.addEventListener("tc:rehearsal-off", stop);
  window.addEventListener("pagehide", stop);
})();

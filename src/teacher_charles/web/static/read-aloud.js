// The read-aloud drill (E12): hear the pair, say it wrong once, read it three
// times, and one card over the session.
//
// ## How a take is captured
//
// Each take starts with a second of the room before the student is told to
// read, and ends on the silence after the speech rather than on a clock. Both
// come from one evening of measurement (plan §5, read-aloud paragraphs): five
// fixed seconds cut a sentence off, and nine diluted a one-second sentence
// until gate 1's contrast question refused it. The second of room is also
// what gate 1 measures the voice against, so the server's signal questions
// run on every take before the recognizer hears it.
//
// The thresholds below are the page's choices, not measurements: they decide
// when to stop listening, and the server decides what the take was worth.
// Whatever they cut wrongly, gate 1's head-cut and truncated warnings catch.
//
// ## Why the browser's processing is off
//
// Noise suppression, echo cancellation and gain control change the sound the
// recognizer hears, and none of them was on in the E9 recordings every
// threshold here was set on. The operating point is part of the data
// (metodo §8), so the page asks for the raw microphone.
//
// The raw samples come from a ScriptProcessorNode and are downsampled to
// 16 kHz here, as follow.js does and for its reasons.

(() => {
  const state = document.getElementById("read-aloud-state");
  if (!state) return;

  const RATE = 16000;
  const BLOCK_S = 0.1;
  const ROOM_S = 1.0;
  // A block this many times the room's level, and above an absolute floor,
  // is speech.
  const SPEECH_OVER_ROOM = 4;
  const SPEECH_RMS_MIN = 0.01;
  // This much quiet after speech ends the take.
  const END_QUIET_S = 0.8;
  // No speech this long after "read now" ends the take with nothing sent.
  const NO_SPEECH_S = 6;
  // The server's ceiling is ten seconds of audio; stop short of it.
  const MAX_S = 9.5;

  const $ = (id) => document.getElementById(id);
  const pick = $("ra-pick");
  const drill = $("ra-drill");
  const cardBox = $("ra-card");
  const status = $("ra-status");
  const takeButton = $("ra-take");
  const deliberateButton = $("ra-deliberate");
  const finishButton = $("ra-finish");
  const dots = $("ra-dots");

  let session = null; // the /api/read-aloud/start answer, plus { judged }
  let busy = false;

  function say(text) {
    status.textContent = text;
  }

  /** Text with *word* as emphasis, built without innerHTML. */
  function rich(el, text) {
    el.textContent = "";
    text.split(/(\*[^*]+\*)/).forEach((part) => {
      if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
        const em = document.createElement("em");
        em.textContent = part.slice(1, -1);
        el.append(em);
      } else {
        el.append(part);
      }
    });
  }

  function rms(samples, from, to) {
    let sum = 0;
    for (let k = from; k < to; k++) sum += samples[k] * samples[k];
    return to > from ? Math.sqrt(sum / (to - from)) : 0;
  }

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

  /**
   * One take: a second of room, then speech until the quiet after it.
   * Resolves to base64 PCM at 16 kHz, or null when nobody spoke.
   */
  async function capture() {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
    const ctx = new AudioContext();
    await ctx.resume().catch(() => {});
    const rate = ctx.sampleRate;
    const block = Math.round(BLOCK_S * rate);
    const source = ctx.createMediaStreamSource(stream);
    const tap = ctx.createScriptProcessor(4096, 1, 1);
    const mute = ctx.createGain();
    mute.gain.value = 0;

    let length = 0;
    const release = () => {
      tap.disconnect();
      source.disconnect();
      mute.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      ctx.close().catch(() => {});
    };

    try {
      return await new Promise((resolve) => {
        let room = null;
        let spokeAt = null;
        let quietSince = null;
        let scanned = 0;
        let flat = null;
        say("Quiet for a second…");
        tap.onaudioprocess = (event) => {
          const data = new Float32Array(event.inputBuffer.getChannelData(0));
          length += data.length;
          if (!flat || flat.length < length) {
            const grown = new Float32Array(Math.max(length, (flat?.length || 0) * 2));
            if (flat) grown.set(flat.subarray(0, length - data.length));
            flat = grown;
          }
          flat.set(data, length - data.length);

          if (room === null) {
            if (length < ROOM_S * rate) return;
            room = rms(flat, 0, Math.round(ROOM_S * rate));
            scanned = Math.round(ROOM_S * rate);
            say("Read now.");
          }
          const threshold = Math.max(SPEECH_OVER_ROOM * room, SPEECH_RMS_MIN);
          while (scanned + block <= length) {
            const loud = rms(flat, scanned, scanned + block) >= threshold;
            const at = scanned / rate;
            if (loud) {
              if (spokeAt === null) spokeAt = at;
              quietSince = null;
            } else if (spokeAt !== null && quietSince === null) {
              quietSince = at;
            }
            scanned += block;
          }
          const now = length / rate;
          const ended = spokeAt !== null && quietSince !== null && now - quietSince >= END_QUIET_S;
          if (ended || now >= MAX_S) {
            resolve(toBase64(downsample(flat.subarray(0, length), rate)));
          } else if (spokeAt === null && now - ROOM_S >= NO_SPEECH_S) {
            resolve(null);
          }
        };
        source.connect(tap);
        tap.connect(mute);
        mute.connect(ctx.destination);
      });
    } finally {
      release();
    }
  }

  async function post(url, body) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    return data;
  }

  function drawDots() {
    dots.textContent = "";
    for (let k = 0; k < session.takes; k++) {
      const dot = document.createElement("span");
      dot.className = k < session.judged ? "ra-dot done" : "ra-dot";
      dot.textContent = k < session.judged ? "●" : "○";
      dots.append(dot);
    }
    takeButton.textContent = `● Read take ${Math.min(session.judged + 1, session.takes)}`;
  }

  function show(which) {
    pick.hidden = which !== "pick";
    drill.hidden = which !== "drill";
    cardBox.hidden = which !== "card";
  }

  async function start(ruleId) {
    if (busy) return;
    busy = true;
    try {
      session = await post("/api/read-aloud/start", ruleId ? { rule_id: ruleId } : {});
      session.judged = 0;
    } catch (error) {
      busy = false;
      alert(`Could not start: ${error.message}`);
      return;
    }
    busy = false;
    $("ra-text").textContent = session.text;
    $("ra-ipa").textContent = session.ipa ? `/${session.ipa}/` : "";
    document.querySelectorAll(".ra-right").forEach((el) => (el.textContent = session.right));
    document.querySelectorAll(".ra-wrong").forEach((el) => (el.textContent = session.wrong));
    $("ra-step-pair").hidden = !session.pair;
    $("ra-step-deliberate").hidden = !session.deliberate;
    $("ra-step-takes").hidden = session.deliberate;
    document.querySelector("#ra-step-takes .ra-n").textContent = session.deliberate ? "3" : session.pair ? "2" : "1";
    drawDots();
    say(session.picked_by === "schedule" ? "Your schedule picked this one." : "");
    show("drill");
  }

  async function take(deliberate) {
    if (busy || !session) return;
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) {
      say("This browser cannot record.");
      return;
    }
    busy = true;
    takeButton.disabled = deliberateButton.disabled = finishButton.disabled = true;
    try {
      const audio = await capture();
      if (audio === null) {
        say("I did not hear anything. Press the button and read when it says so.");
        return;
      }
      say("Listening back…");
      const result = await post("/api/read-aloud/take", {
        session_id: session.session_id,
        deliberate,
        audio,
      });
      if (deliberate) {
        say(result.note || "");
        if (result.recorded) {
          $("ra-step-deliberate").hidden = true;
          $("ra-step-takes").hidden = false;
        }
        return;
      }
      if (!result.recorded) {
        // Gate 1 or 2: capture advice, never pronunciation. Read it again.
        say(result.advice || "I could not use that take. Read it once more.");
        return;
      }
      session.judged = result.judged;
      drawDots();
      if (session.judged >= session.takes) {
        await finish();
      } else {
        say(`Take ${session.judged} of ${session.takes} done.`);
      }
    } catch (error) {
      say(`That take failed: ${error.message}`);
    } finally {
      busy = false;
      takeButton.disabled = deliberateButton.disabled = finishButton.disabled = false;
    }
  }

  async function finish() {
    if (!session) return;
    const id = session.session_id;
    session = null;
    try {
      const result = await post("/api/read-aloud/finish", { session_id: id });
      const lines = $("ra-lines");
      lines.textContent = "";
      result.card.lines.forEach((line) => {
        const p = document.createElement("p");
        rich(p, line);
        lines.append(p);
      });
      say("");
      show("card");
    } catch (error) {
      say(`Could not finish: ${error.message}`);
    }
  }

  document.querySelectorAll(".ra-practise").forEach((button) => {
    button.addEventListener("click", () => start(button.dataset.rule));
  });
  $("ra-schedule").addEventListener("click", () => start(null));
  document.querySelectorAll("#ra-step-pair button").forEach((button) => {
    button.addEventListener("click", () => {
      if (!session) return;
      const url = `/api/read-aloud/pair/${encodeURIComponent(session.rule_id)}/${button.dataset.side}?phrase=${session.phrase}`;
      new Audio(url).play().catch((error) => say(`Could not play: ${error.message}`));
    });
  });
  deliberateButton.addEventListener("click", () => take(true));
  takeButton.addEventListener("click", () => take(false));
  finishButton.addEventListener("click", () => {
    if (!busy) finish();
  });
  $("ra-again").addEventListener("click", () => window.location.reload());
})();

"use strict";

// As duas pontas de voz, no navegador.
//
// Publica `window.CharlesVoice`. Carregado ANTES do app.js, que chama
// `CharlesVoice.speak()` nos quadros `sentence` do /api/chat. Sem build step,
// a ordem das tags <script> e a dependencia -- esta e a razao de a pagina
// carregar os dois nessa ordem.
//
// ## Por que a transcricao vai para o campo de texto
//
// O STT erra. Um erro de STT que vira turno faz o Charles corrigir um erro que
// o aluno nao cometeu -- o pior defeito possivel neste produto, e exatamente o
// que a base curada da E3/E4 existe para evitar do lado do modelo. Com a
// transcricao no campo, o aluno le, conserta uma palavra e manda. E um
// pressionar de tecla a mais, e compra a unica salvaguarda que existe contra
// um pipeline de audio que nao tem como estar sempre certo.
//
// ## Por que a fila toca por indice, e nao por ordem de chegada
//
// As sentencas saem do servidor em ordem, mas cada uma e um fetch proprio para
// o /api/tts, e fetch nao garante ordem de resolucao. Tocar por ordem de
// chegada embaralharia a fala do professor na primeira vez que a rede
// hesitasse. Os pedidos saem todos de uma vez (e assim a sintese da sentenca 2
// corre dentro da reproducao da 1); a reproducao espera o indice certo.

const CharlesVoice = (() => {
  const form = document.getElementById("composer");
  const enabled = form?.dataset.voice === "on";

  const talkButton = document.getElementById("talk");
  const speakToggle = document.getElementById("speak");

  // Lembrar a escolha entre recarregamentos: trocar de modo e trocar de perfil
  // recarregam a pagina, e perder o alto-falante a cada troca seria irritante.
  const SPEAK_KEY = "charles.speak";
  if (speakToggle) {
    const stored = window.localStorage?.getItem(SPEAK_KEY);
    if (stored !== null && stored !== undefined) speakToggle.checked = stored === "1";
    speakToggle.addEventListener("change", () => {
      window.localStorage?.setItem(SPEAK_KEY, speakToggle.checked ? "1" : "0");
      if (!speakToggle.checked) stopSpeaking();
    });
  }

  // --- callbacks que o app.js preenche ---------------------------------- //

  let onStatus = () => {};
  let onTranscript = () => {};

  // --- reproducao -------------------------------------------------------- //

  // indice -> Promise<Blob>. Os fetches comecam na hora; a reproducao espera.
  let pending = new Map();
  // Identifica o turno em curso. Invalidar o turno faz o laco de reproducao
  // desistir sozinho, sem flag separada: ele compara a cada passo.
  let playingTurn = 0;
  let current = null;
  // `done` chegou: nao vem mais sentenca, entao a fila pode parar de esperar
  // em vez de girar para sempre no indice que nunca vai existir.
  let finished = true;

  function stopSpeaking() {
    playingTurn += 1;
    pending = new Map();
    if (current) {
      current.pause();
      current = null;
    }
  }

  /** Pede a sintese de uma sentenca. Nao espera: a fila cuida da ordem. */
  function speak(index, text) {
    if (!enabled || !speakToggle?.checked || !text) return;

    const turn = playingTurn;
    pending.set(
      index,
      fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      }).then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.blob();
      })
    );

    if (index === 0) playQueue(turn);
  }

  /** Toca da sentenca 0 em diante, esperando cada indice chegar. */
  async function playQueue(turn) {
    let index = 0;
    while (turn === playingTurn) {
      const waiting = pending.get(index);
      if (!waiting) {
        // A sentenca ainda nao existe. Se o turno acabou, a fila acabou.
        if (finished) return;
        await new Promise((resolve) => setTimeout(resolve, 60));
        continue;
      }

      let blob;
      try {
        blob = await waiting;
      } catch (error) {
        onStatus(`Charles lost his voice: ${error.message}`, true);
        return;
      }
      if (turn !== playingTurn) return;

      try {
        await playBlob(blob);
      } catch {
        // Autoplay bloqueado, ou aba sem permissao de som. Nao e erro de
        // turno -- o texto esta na tela -- e insistir a cada sentenca encheria
        // a barra de status de ruido.
        onStatus("Charles is speaking, but the browser blocked the sound.", true);
        return;
      }
      pending.delete(index);
      index += 1;
    }
  }

  function playBlob(blob) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      current = audio;
      const release = () => {
        URL.revokeObjectURL(url);
        if (current === audio) current = null;
      };
      audio.addEventListener("ended", () => {
        release();
        resolve();
      });
      audio.addEventListener("error", () => {
        release();
        reject(new Error("audio error"));
      });
      audio.play().catch((error) => {
        release();
        reject(error);
      });
    });
  }

  function startTurn() {
    stopSpeaking();
    finished = false;
  }

  function endTurn() {
    finished = true;
  }

  // --- gravacao ---------------------------------------------------------- //

  // Opus em webm (Chromium) ou em ogg (Firefox). O PyAV decodifica os dois, e
  // deixar o navegador escolher evita transcodificar nada aqui.
  const MIME_TYPES = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
  ];

  // Soltar o botao antes disto e um clique, nao um "segura e fala". Nesse caso
  // a gravacao fica travada ligada ate o clique seguinte -- segurar o mouse
  // por vinte segundos e desconfortavel, e em telas de toque e pior.
  const LATCH_MS = 250;

  let recorder = null;
  let chunks = [];
  let startedAt = 0;
  let latched = false;
  let stream = null;

  function supported() {
    return Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
  }

  function pickMimeType() {
    for (const type of MIME_TYPES) {
      if (window.MediaRecorder.isTypeSupported(type)) return type;
    }
    return "";
  }

  async function startRecording() {
    if (recorder || !supported()) return;

    // A fala do professor entraria no microfone junto com a do aluno.
    stopSpeaking();

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
    } catch (error) {
      onStatus(`No microphone: ${error.message}`, true);
      return;
    }

    const mimeType = pickMimeType();
    recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    chunks = [];
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    });
    recorder.addEventListener("stop", transcribe);
    recorder.start();
    startedAt = Date.now();
    talkButton?.classList.add("recording");
    onStatus("Listening…");
  }

  function stopRecording() {
    if (!recorder) return;
    recorder.stop();
    recorder = null;
    for (const track of stream?.getTracks() ?? []) track.stop();
    stream = null;
    talkButton?.classList.remove("recording");
    latched = false;
  }

  async function transcribe() {
    const audio = new Blob(chunks, { type: chunks[0]?.type || "audio/webm" });
    chunks = [];
    if (audio.size < 1024) {
      onStatus("That was too short — hold the button while you speak.");
      return;
    }

    onStatus("Writing down what you said…");
    try {
      const response = await fetch("/api/stt", {
        method: "POST",
        headers: { "Content-Type": audio.type },
        body: audio,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);

      if (!data.text) {
        onStatus("Charles did not hear any words. Try again, a bit closer.");
        return;
      }
      onTranscript(data.text);
      onStatus(`Heard you in ${(data.elapsed_ms / 1000).toFixed(1)} s — fix it if it is wrong.`);
    } catch (error) {
      onStatus(`Could not transcribe: ${error.message}`, true);
    }
  }

  if (talkButton) {
    if (!supported()) {
      talkButton.disabled = true;
      talkButton.title = "This browser cannot record audio";
    } else {
      talkButton.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        if (latched) {
          stopRecording();
          return;
        }
        startRecording();
      });

      const release = () => {
        if (!recorder) return;
        if (Date.now() - startedAt < LATCH_MS) {
          // Clique curto: trava ligado ate o proximo clique.
          latched = true;
          talkButton.classList.add("latched");
          onStatus("Listening — click again when you are done.");
          return;
        }
        talkButton.classList.remove("latched");
        stopRecording();
      };

      talkButton.addEventListener("pointerup", release);
      talkButton.addEventListener("pointercancel", release);
      // Arrastar o dedo/cursor fora do botao sem soltar: trata como soltar, em
      // vez de deixar o microfone ligado sem ninguem perceber.
      talkButton.addEventListener("pointerleave", () => {
        if (recorder && !latched) release();
      });
    }
  }

  return {
    enabled,
    speak,
    startTurn,
    endTurn,
    stopSpeaking,
    /** O app.js injeta como mostrar status e o que fazer com a transcricao. */
    wire({ status, transcript }) {
      if (status) onStatus = status;
      if (transcript) onTranscript = transcript;
    },
    get speaking() {
      return Boolean(current);
    },
  };
})();

window.CharlesVoice = CharlesVoice;

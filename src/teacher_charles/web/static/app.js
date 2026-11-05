"use strict";

// Cliente do chat.
//
// Le a resposta com fetch + reader de stream, nao com EventSource: aquele so
// faz GET, e a fala do aluno e dado pessoal que nao deve ir em query string.
// Cada pedaco e escrito na tela assim que chega -- acumular aqui anularia o
// streaming, que e o que torna 7-8 tok/s tolerável.

const transcript = document.getElementById("transcript");
const form = document.getElementById("composer");
const input = document.getElementById("message");
const sendButton = document.getElementById("send");
const statusLine = document.getElementById("status");
const resetButton = document.getElementById("reset");

let busy = false;

function addTurn(role, text) {
  const article = document.createElement("article");
  article.className = `turn ${role}`;

  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "student" ? "You" : "Charles";

  const what = document.createElement("div");
  what.className = "what";
  // textContent, nunca innerHTML: o texto vem do modelo e do aluno.
  what.textContent = text;

  article.append(who, what);
  transcript.append(article);
  scrollToEnd();
  return what;
}

function scrollToEnd() {
  transcript.scrollTop = transcript.scrollHeight;
}

function setBusy(value) {
  busy = value;
  sendButton.disabled = value;
  input.disabled = value;
  if (!value) input.focus();
}

function setStatus(text, isError = false) {
  statusLine.textContent = text;
  statusLine.classList.toggle("error", isError);
}

// Cresce o textarea com o texto, ate um limite.
function autoGrow() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}
input.addEventListener("input", autoGrow);

// Enter envia; Shift+Enter quebra linha.
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

/**
 * Le um corpo SSE pedaco por pedaco e chama onFrame para cada evento.
 *
 * Um chunk da rede pode cortar um evento no meio, entao o que sobra fica no
 * buffer para o proximo. Sem isso, uma resposta longa perde eventos.
 */
async function readFrames(response, onFrame) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let separator;
    while ((separator = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, separator).trim();
      buffer = buffer.slice(separator + 2);
      if (!raw.startsWith("data:")) continue;
      const payload = raw.slice(5).trim();
      if (!payload) continue;
      try {
        onFrame(JSON.parse(payload));
      } catch {
        // Evento ilegivel: ignora em vez de derrubar a conversa inteira.
      }
    }
  }
}

async function sendMessage(text) {
  addTurn("student", text);
  const target = addTurn("teacher", "");
  target.classList.add("streaming");
  setStatus("Charles is thinking…");

  let gotText = false;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`HTTP ${response.status}`);
    }

    await readFrames(response, (frame) => {
      if (frame.type === "chunk") {
        if (!gotText) {
          setStatus("");
          gotText = true;
        }
        target.textContent += frame.text;
        scrollToEnd();
      } else if (frame.type === "done") {
        setStatus(
          `first word ${frame.first_chunk_ms} ms · turn ${(frame.total_ms / 1000).toFixed(1)} s`
        );
      } else if (frame.type === "error") {
        target.classList.add("failed");
        if (!gotText) target.textContent = "—";
        setStatus(frame.message, true);
      }
    });
  } catch (error) {
    target.classList.add("failed");
    if (!gotText) target.textContent = "—";
    setStatus(`Could not reach the teacher: ${error.message}`, true);
  } finally {
    target.classList.remove("streaming");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;

  const text = input.value.trim();
  if (!text) return;

  input.value = "";
  autoGrow();
  setBusy(true);
  try {
    await sendMessage(text);
  } finally {
    setBusy(false);
  }
});

resetButton.addEventListener("click", async () => {
  if (busy) return;
  setBusy(true);
  try {
    await fetch("/api/reset", { method: "POST" });
    transcript.replaceChildren();
    addTurn("teacher", "Hello! I'm Charles. Tell me about your day — in English.");
    setStatus("");
  } catch (error) {
    setStatus(`Could not start a new conversation: ${error.message}`, true);
  } finally {
    setBusy(false);
  }
});

input.focus();
scrollToEnd();

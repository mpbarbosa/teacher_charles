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
const shutdownButton = document.getElementById("shutdown");
const levelSelect = document.getElementById("level");
const levelNote = document.getElementById("level-note");
const nameInput = document.getElementById("student-name");
const modeButtons = document.querySelectorAll("button.mode");
// Pode nao existir: a fonetica e desligavel, e sem ela o cabecalho nao
// desenha o interruptor.
const ipaToggle = document.getElementById("show-ipa");

// Pode nao existir: a voz e opcional, e sem ela a pagina nem carrega voice.js.
// Dai o objeto inerte em vez de um `if` em cada ponto de uso.
const voice = window.CharlesVoice ?? {
  enabled: false,
  speak() {},
  startTurn() {},
  endTurn() {},
  stopSpeaking() {},
  stopListening() {},
  wire() {},
};

let busy = false;

function addTurn(role, text, ipa = "") {
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
  const bundle = { article, what };
  if (ipa) setIpa(bundle, ipa);
  scrollToEnd();
  return bundle;
}

/**
 * Pendura (ou atualiza) a linha de pronuncia desta bolha.
 *
 * Criada so quando ha IPA de verdade: uma linha vazia embaixo de cada fala
 * seria um espaco em branco que o aluno nao sabe ler. E ela entra DEPOIS da
 * fala e ANTES das correcoes -- a correcao anota o que o aluno disse, e ficaria
 * longe demais do texto com o IPA no meio.
 */
function setIpa(bundle, ipa) {
  if (!ipa) return;
  if (!bundle.ipa) {
    bundle.ipa = document.createElement("div");
    bundle.ipa.className = "ipa";
    // `en-fonipa`: diz ao navegador (e ao leitor de tela) que isto e notacao
    // fonetica, nao ingles para ser lido em voz alta.
    bundle.ipa.lang = "en-fonipa";
    bundle.ipa.setAttribute("aria-label", "Pronunciation");
    bundle.what.after(bundle.ipa);
  }
  bundle.ipa.textContent = ipa;
  scrollToEnd();
}

/**
 * Anexa as correcoes ao turno do aluno.
 *
 * A correcao e dado estruturado, nao texto do professor: ela anota o que o
 * aluno disse. `rule_id` presente significa que a explicacao veio da base
 * curada, nao da improvisacao do modelo -- e a UI mostra isso.
 */
function addCorrections(article, errors) {
  if (!errors || errors.length === 0) return;

  const box = document.createElement("div");
  box.className = "corrections";

  for (const error of errors) {
    const item = document.createElement("div");
    item.className = error.rule_id ? "correction curated" : "correction";

    const fix = document.createElement("div");
    fix.className = "fix";
    const was = document.createElement("s");
    was.textContent = error.span;
    const now = document.createElement("strong");
    now.textContent = error.correction;
    fix.append(was, document.createTextNode(" "), now);

    const why = document.createElement("div");
    why.className = "why";
    why.textContent = error.explanation;

    item.append(fix, why);

    if (error.rule_id) {
      const rule = document.createElement("div");
      rule.className = "rule";
      rule.textContent = error.rule_id;
      item.append(rule);
    }

    box.append(item);
  }

  article.append(box);
  scrollToEnd();
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

// Fim de expediente: a conversa continua na tela para ser lida, e nenhum
// controle continua prometendo um servidor que nao esta mais la. `busy` fica
// ligado, entao os outros handlers tambem saem na primeira linha.
function freeze() {
  document.body.classList.add("stopped");
  // O microfone tem que fechar aqui, e nao so o alto-falante: o botao de falar
  // fica desabilitado logo abaixo, mas uma gravacao travada continuaria aberta
  // atras de uma pagina que ja disse que o Charles parou -- e sem nenhum
  // /api/stt de pe para onde mandar o audio.
  voice.stopSpeaking();
  voice.stopListening();
  for (const control of document.querySelectorAll("button, input, select, textarea")) {
    control.disabled = true;
  }
  // `pointer-events: none` nao tira a ancora do Tab: um usuario de teclado
  // ainda chegaria no link do error log, e o Enter sairia da pagina congelada
  // para um servidor que parou. Sem `href` ela deixa de ser focavel e de
  // navegar; a classe no body so cuida da aparencia.
  for (const link of document.querySelectorAll("a.button")) {
    link.removeAttribute("href");
  }
  const notice = document.createElement("div");
  notice.className = "notice";
  notice.textContent =
    "Charles has stopped. You can close this tab — scripts/run.sh brings him back.";
  transcript.append(notice);
  scrollToEnd();
  setStatus("");
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
  const student = addTurn("student", text);
  const teacher = addTurn("teacher", "");
  const target = teacher.what;
  target.classList.add("streaming");
  setStatus("Charles is thinking…");

  let gotText = false;
  // Zera a fila de audio do turno anterior antes de o novo comecar a chegar.
  voice.startTurn();

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
      } else if (frame.type === "phonetics") {
        // Duas por turno: a do aluno chega antes do primeiro pedaco do
        // professor (o G2P nao espera modelo nenhum), a do professor depois do
        // ultimo. O papel diz em qual bolha pendurar -- a ordem nao serve,
        // porque a analise pode entrar entre as duas.
        setIpa(frame.role === "student" ? student : teacher, frame.ipa);
      } else if (frame.type === "analysis") {
        // Chega em geral antes de o professor terminar de falar: as duas
        // etapas do turno rodam em paralelo.
        addCorrections(student.article, frame.errors);
      } else if (frame.type === "sentence") {
        // Uma sentenca fechou. O pedido de sintese sai agora, e corre
        // enquanto o modelo escreve a proxima -- e disso que o TTS
        // incremental vive.
        voice.speak(frame.index, frame.text);
      } else if (frame.type === "level") {
        applyLevel(frame);
      } else if (frame.type === "done") {
        const analysis = frame.analysis_ms
          ? ` · analysis ${(frame.analysis_ms / 1000).toFixed(1)} s`
          : "";
        setStatus(
          `first word ${frame.first_chunk_ms} ms · turn ` +
            `${(frame.total_ms / 1000).toFixed(1)} s${analysis}`
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
    // Nao vem mais sentenca: a fila para de esperar pelo indice seguinte.
    voice.endTurn();
  }
}

/**
 * Mostra o nivel em vigor depois do turno.
 *
 * O frame chega em TODO turno, nao so quando o nivel muda: o badge tem que
 * estar certo depois de um reinicio, nao so depois de uma mudanca. A nota em
 * ingles aparece apenas na mudanca -- o aluno merece saber por que o Charles
 * passou a falar diferente com ele.
 */
function applyLevel(frame) {
  if (!levelSelect) return;
  levelSelect.value = frame.level;
  levelSelect.dataset.source = frame.changed ? "auto" : levelSelect.dataset.source;
  if (frame.changed) {
    const direction = frame.level > frame.previous ? "up" : "down";
    levelNote.textContent = `Charles moved you ${direction} to ${frame.level}.`;
    levelNote.classList.add("changed");
  }
}

async function postProfile(body) {
  const response = await fetch("/api/profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

if (levelSelect) {
  levelSelect.addEventListener("change", async () => {
    const chosen = levelSelect.value;
    try {
      const profile = await postProfile({ level: chosen });
      levelSelect.value = profile.level;
      levelSelect.dataset.source = profile.level_source;
      levelNote.textContent = `Level set to ${profile.level}.`;
      levelNote.classList.remove("changed");
    } catch (error) {
      setStatus(`Could not change your level: ${error.message}`, true);
    }
  });
}

if (nameInput) {
  // Trocar o nome troca de aluno, e o historico da tela e do outro perfil:
  // recarregar e mais honesto que manter a conversa de outra pessoa na tela.
  const switchProfile = async () => {
    const name = nameInput.value.trim();
    if (!name || name === nameInput.defaultValue) return;
    try {
      await postProfile({ name });
      window.location.reload();
    } catch (error) {
      setStatus(`Could not switch profile: ${error.message}`, true);
    }
  };
  nameInput.addEventListener("change", switchProfile);
  nameInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      switchProfile();
    }
  });
}

// Trocar de modo e trocar de atividade: a sessao muda, e com ela a
// transcricao. Recarregar e mais honesto que manter na tela a conversa de
// outro modo.
for (const button of modeButtons) {
  button.addEventListener("click", async () => {
    if (busy || button.classList.contains("active")) return;
    setBusy(true);
    try {
      const response = await fetch("/api/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: button.dataset.mode }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      window.location.reload();
    } catch (error) {
      setBusy(false);
      setStatus(`Could not switch mode: ${error.message}`, true);
    }
  });
}

// --- O interruptor da pronuncia -------------------------------------------- //
//
// So esconde: o IPA ja veio com a pagina e com o turno, entao voltar a
// mostra-lo nao custa pedido nenhum. A escolha fica no navegador porque e
// preferencia de leitura desta tela, nao dado do aluno -- nao tem por que
// ocupar uma coluna no banco nem viajar pela rede.
const IPA_PREFERENCE = "charles:show-ipa";

function applyIpaPreference(show) {
  document.body.classList.toggle("no-ipa", !show);
}

if (ipaToggle) {
  let stored = null;
  try {
    stored = window.localStorage.getItem(IPA_PREFERENCE);
  } catch {
    // Navegador com armazenamento bloqueado: segue no default, que e mostrar.
  }
  ipaToggle.checked = stored !== "off";
  applyIpaPreference(ipaToggle.checked);

  ipaToggle.addEventListener("change", () => {
    applyIpaPreference(ipaToggle.checked);
    try {
      window.localStorage.setItem(IPA_PREFERENCE, ipaToggle.checked ? "on" : "off");
    } catch {
      // Sem armazenamento a escolha vale so para esta visita. Melhor que
      // derrubar o manipulador inteiro.
    }
  });
}

voice.wire({
  status: setStatus,
  transcript: (text) => {
    // No campo, nunca direto no turno. O STT erra, e um erro de STT que vira
    // turno faz o Charles corrigir um erro que o aluno nao cometeu -- ver o
    // cabecalho de voice.js.
    input.value = input.value.trim() ? `${input.value.trim()} ${text}` : text;
    autoGrow();
    input.focus();
    // Cursor no fim, para o aluno conseguir corrigir a ultima palavra, que e
    // onde o STT mais erra.
    input.setSelectionRange(input.value.length, input.value.length);
  },
});

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
  // Quem pede conversa nova nao quer ouvir o fim da anterior.
  voice.stopSpeaking();
  try {
    await fetch("/api/reset", { method: "POST" });
    transcript.replaceChildren();
    // A saudacao e o IPA dela vem do servidor pelos `data-` do <main>: o
    // espeak roda la, e repetir a frase aqui a deixaria sem transcricao.
    addTurn("teacher", transcript.dataset.greeting ?? "", transcript.dataset.greetingIpa ?? "");
    setStatus("");
  } catch (error) {
    setStatus(`Could not start a new conversation: ${error.message}`, true);
  } finally {
    setBusy(false);
  }
});

// Desligar nao tem desfazer pela pagina -- o Charles so volta pelo terminal --,
// e o botao fica a um centimetro do "new conversation", que nao perde nada. Dai
// a confirmacao, que o reset ao lado nao precisa ter.
shutdownButton.addEventListener("click", async () => {
  if (busy) return;
  if (!window.confirm("Stop Charles? He only comes back from the terminal.")) return;

  voice.stopSpeaking();
  setBusy(true);
  setStatus("Stopping Charles…");
  try {
    const response = await fetch("/api/shutdown", { method: "POST" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    // O servidor responde antes de parar, de proposito: sem isso o navegador
    // mostraria conexao cortada no lugar da despedida.
    freeze();
  } catch (error) {
    setBusy(false);
    setStatus(`Could not stop Charles: ${error.message}`, true);
  }
});

input.focus();
scrollToEnd();

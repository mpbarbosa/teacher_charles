"use strict";

// The phonetics page: a sentence goes in, its IPA comes out.
//
// The sentence is POSTed to /api/phonetics (never put in the URL: it is what
// the student typed). The answer is one entry per line of the box, each a list
// of sentences with the whole sentence's IPA and its words paired with their
// own IPA. Words espeak runs together ("on the" -> ɔnðə) arrive as one pair.
// Everything is drawn with textContent, so nothing typed becomes markup.

(() => {
  const form = document.getElementById("phonetics-form");
  const input = document.getElementById("phonetics-text");
  const result = document.getElementById("phonetics-result");
  if (!form || !input || !result) return;
  const submit = form.querySelector("button[type=submit]");

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
    return block;
  }

  function show(body) {
    result.replaceChildren();
    for (const line of body.lines) {
      for (const sentence of line) result.append(sentenceBlock(sentence));
    }
    if (!result.childElementCount) result.append(el("p", "lead empty", "Nothing to transcribe."));
    result.hidden = false;
  }

  function fail(message) {
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

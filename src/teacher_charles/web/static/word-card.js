"use strict";

// The card that opens when the pointer rests on a word of a lesson's full
// text: the word's senses, each with an English definition, an example and
// the Portuguese for it, from WordNet through POST /api/define
// (teaching/dictionary.py) — never from the model.
//
// Loaded only when the lookup file is on this machine (the template leaves
// the script out otherwise), so a hover never meets a 503 in normal use.
//
// Hover only, on purpose. A click on a word already starts the reading from
// there (lesson.js), and a touch has no hover: a tap that both opened a card
// and started the voice would do two things nobody asked for at once.

(() => {
  const reading = document.getElementById("full-text-reading");
  if (!reading) return;

  // Long enough that sweeping the pointer across a line opens nothing; short
  // enough that resting on a word feels like asking.
  const OPEN_DELAY_MS = 350;
  // Time to move from the word into the card without it closing.
  const CLOSE_DELAY_MS = 200;
  const GAP_PX = 6;
  const MARGIN_PX = 8;

  const card = document.createElement("div");
  card.className = "word-card";
  card.id = "word-card";
  card.setAttribute("role", "tooltip");
  card.hidden = true;
  document.body.append(card);

  /** word (lower case) -> Promise<entry>, for the page's lifetime. */
  const looked = new Map();
  let openTimer = 0;
  let closeTimer = 0;
  let current = null; // the .w the card is for, or is about to be for

  function lookup(text) {
    const key = text.toLowerCase();
    if (!looked.has(key)) {
      const request = fetch("/api/define", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ word: text }),
      }).then(async (response) => {
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
        return body;
      });
      // A failure is not kept: the next hover asks again.
      request.catch(() => looked.delete(key));
      looked.set(key, request);
    }
    return looked.get(key);
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  /** The entry as DOM. Text only, never HTML: definitions come from a file. */
  function render(entry, written) {
    const parts = [];
    if (!entry.found) {
      parts.push(el("div", "wc-head", null));
      parts[0].append(el("span", "wc-lemma", written));
      parts.push(el("p", "wc-miss", entry.note));
      return parts;
    }
    for (const headword of entry.headwords) {
      const head = el("div", "wc-head");
      head.append(el("span", "wc-lemma", headword.lemma), " ", el("span", "wc-pos", headword.pos));
      if (headword.ipa) head.append(" ", el("span", "wc-ipa", `/${headword.ipa}/`));
      parts.push(head);

      const senses = el("ol", "wc-senses");
      for (const sense of headword.senses) {
        const item = el("li");
        item.append(el("span", "wc-def", sense.definition));
        if (sense.example) item.append(el("span", "wc-ex", `“${sense.example}”`));
        if (sense.portuguese.length) {
          const pt = el("span", "wc-pt");
          pt.append(el("abbr", "wc-lang", "PT"));
          pt.lastChild.title = "Portuguese";
          const words = el("span", null, sense.portuguese.join(", "));
          words.lang = "pt";
          pt.append(" ", words);
          item.append(pt);
        }
        senses.append(item);
      }
      parts.push(senses);
    }
    if (entry.more) {
      parts.push(el("p", "wc-more", `+${entry.more} more sense${entry.more === 1 ? "" : "s"}`));
    }
    parts.push(el("p", "wc-note", entry.note));
    const sources = el("p", "wc-src");
    entry.sources.forEach((source, i) => {
      if (i) sources.append(" · ");
      const link = el("a", null, source.name);
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener";
      link.title = source.licence;
      sources.append(link);
    });
    parts.push(sources);
    return parts;
  }

  /**
   * Below the word, or above it when only above has room. When neither side
   * holds the whole card, it takes the roomier one and scrolls inside: a card
   * running off the viewport hides the senses it was opened for.
   */
  function place(word) {
    const box = word.getBoundingClientRect();
    card.style.left = "0px";
    card.style.top = "0px";
    card.style.maxHeight = "";
    const { width, height } = card.getBoundingClientRect();
    const maxLeft = window.innerWidth - width - MARGIN_PX;
    const left = Math.max(MARGIN_PX, Math.min(box.left, maxLeft));
    const roomBelow = window.innerHeight - MARGIN_PX - (box.bottom + GAP_PX);
    const roomAbove = box.top - GAP_PX - MARGIN_PX;
    const below = height <= roomBelow || (height > roomAbove && roomBelow >= roomAbove);
    const shown = Math.min(height, below ? roomBelow : roomAbove);
    if (shown < height) card.style.maxHeight = `${shown}px`;
    card.style.left = `${left}px`;
    card.style.top = `${below ? box.bottom + GAP_PX : box.top - GAP_PX - shown}px`;
  }

  function show(word, parts) {
    card.replaceChildren(...parts);
    card.hidden = false;
    place(word);
    word.setAttribute("aria-describedby", card.id);
  }

  function close() {
    clearTimeout(openTimer);
    clearTimeout(closeTimer);
    if (current) current.removeAttribute("aria-describedby");
    current = null;
    card.hidden = true;
  }

  async function open(word) {
    const written = word.textContent.trim();
    // A dash or a lone quote is a `.w` too; there is nothing to look up.
    if (!/[\p{L}\p{N}]/u.test(written)) return;
    let parts;
    try {
      parts = render(await lookup(written), written);
    } catch (error) {
      parts = [el("p", "wc-miss", `The dictionary did not answer: ${error.message}`)];
    }
    // The pointer may have moved on while the lookup ran.
    if (current === word) show(word, parts);
  }

  reading.addEventListener("pointerover", (event) => {
    if (event.pointerType && event.pointerType !== "mouse" && event.pointerType !== "pen") return;
    // Written words only: the IPA line's words are `.w` too, under `.ipa`.
    const word = event.target.closest?.(".sentence .w");
    if (!word || !reading.contains(word)) return;
    clearTimeout(closeTimer);
    if (word === current) return;
    clearTimeout(openTimer);
    if (current && !card.hidden) {
      // Moving along the line with a card open: follow at once.
      card.hidden = true;
      current.removeAttribute("aria-describedby");
      current = word;
      open(word);
      return;
    }
    current = word;
    openTimer = setTimeout(() => open(word), OPEN_DELAY_MS);
  });

  reading.addEventListener("pointerout", (event) => {
    const word = event.target.closest?.(".sentence .w");
    if (!word || word !== current || word.contains(event.relatedTarget)) return;
    clearTimeout(openTimer);
    closeTimer = setTimeout(close, CLOSE_DELAY_MS);
  });

  card.addEventListener("pointerenter", () => clearTimeout(closeTimer));
  card.addEventListener("pointerleave", () => {
    closeTimer = setTimeout(close, CLOSE_DELAY_MS);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !card.hidden) close();
  });
  // Fixed to the viewport, so it would drift off its word on scroll.
  window.addEventListener("scroll", () => card.hidden || close(), { passive: true });
  window.addEventListener("resize", () => card.hidden || close());
})();

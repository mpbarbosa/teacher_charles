"use strict";

// The card that opens when the pointer rests on a word — of a lesson's full
// text, of the quote beside one of its points, or of a chat message: the
// word's senses, each with an English definition, an example and the
// Portuguese for it, from WordNet through POST /api/define
// (teaching/dictionary.py) — never from the model.
//
// Loaded only when the lookup file is on this machine (the templates leave
// the script out otherwise), so a hover never meets a 503 in normal use.
//
// Hover only, on purpose. A click on a word of the full text already starts
// the reading from there (lesson.js), and a touch has no hover: a tap that
// both opened a card and started the voice would do two things nobody asked
// for at once.

(() => {
  const reading = document.getElementById("full-text-reading");
  const quotes = [...document.querySelectorAll(".point .from-text")];
  const transcript = document.getElementById("transcript");
  const roots = [reading, ...quotes, transcript].filter(Boolean);
  if (!roots.length) return;
  // Written words only: the full text's IPA line has `.w` words too, under
  // `.ipa`, and those are symbols, not words to look up.
  const WORD = ".sentence .w, .from-text .w";
  // A chat message's own text. Not its IPA line, its corrections or the
  // Structure panel: those are the app's words about the message.
  const BUBBLE = ".turn .what";

  /**
   * Cuts a point's quote into one `.w` per word, as the template cuts the
   * full text. Here and not in the template because the quote is runs of
   * text with the citation in a `<mark>` — or a bare `<q>` on a machine
   * with no copy — and the server's markup is what the lesson tests read
   * back. Only text nodes are split, so the marks stay where they were.
   */
  function cutIntoWords(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      if (!node.data.trim()) continue;
      const pieces = node.data.split(/(\s+)/).filter(Boolean);
      node.replaceWith(
        ...pieces.map((piece) => {
          if (/^\s+$/.test(piece)) return piece;
          const word = document.createElement("span");
          word.className = "w";
          word.textContent = piece;
          return word;
        }),
      );
    }
  }
  quotes.forEach(cutIntoWords);

  // --- What is under the pointer ------------------------------------------ //
  //
  // A target is a word as a text range: `node` and `start` say which, `text`
  // is what gets looked up, `range` is what the card is placed against, and
  // `owner` is the element that carries `aria-describedby` while it is open.

  function spanTarget(word) {
    const range = document.createRange();
    range.selectNodeContents(word);
    return { node: word, start: 0, text: word.textContent.trim(), range, owner: word };
  }

  function caretAt(x, y) {
    if (document.caretPositionFromPoint) {
      const p = document.caretPositionFromPoint(x, y);
      return p && { node: p.offsetNode, offset: p.offset };
    }
    const r = document.caretRangeFromPoint?.(x, y);
    return r && { node: r.startContainer, offset: r.startOffset };
  }

  /**
   * The word of a chat message under the pointer, found from the caret.
   *
   * A message is not cut into spans like the lesson's text, and on purpose:
   * the conversation is an `aria-live` region, and replacing a finished
   * bubble's text with spans reads to a screen reader as new content — it
   * would say Charles's whole answer a second time. Read from the caret,
   * the bubble's DOM is never touched, and a reply still streaming in is as
   * hoverable as a finished one.
   */
  function bubbleTarget(event) {
    const what = event.target.closest?.(BUBBLE);
    if (!what || !transcript?.contains(what)) return null;
    const caret = caretAt(event.clientX, event.clientY);
    if (!caret || caret.node.nodeType !== Node.TEXT_NODE || !what.contains(caret.node)) {
      return null;
    }
    const text = caret.node.data;
    let start = caret.offset;
    let end = caret.offset;
    while (start > 0 && !/\s/.test(text[start - 1])) start--;
    while (end < text.length && !/\s/.test(text[end])) end++;
    if (start === end) return null;
    const range = document.createRange();
    range.setStart(caret.node, start);
    range.setEnd(caret.node, end);
    // The caret snaps to the nearest position, also from the blank beside a
    // short line or the space between two words: only a pointer actually on
    // the word's glyphs counts.
    const { clientX: x, clientY: y } = event;
    const onIt = [...range.getClientRects()].some(
      (r) => x >= r.left && x <= r.right && y >= r.top && y <= r.bottom,
    );
    if (!onIt) return null;
    return { node: caret.node, start, text: text.slice(start, end), range, owner: what };
  }

  function targetAt(event) {
    const word = event.target.closest?.(WORD);
    if (word && roots.some((root) => root.contains(word))) return spanTarget(word);
    return bubbleTarget(event);
  }

  const same = (a, b) => Boolean(a && b && a.node === b.node && a.start === b.start);

  // --- The card ------------------------------------------------------------ //

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
  let current = null; // the target the card is for, or is about to be for

  function lookup(text) {
    const key = text.toLowerCase();
    if (!looked.has(key)) {
      // A body, not a query string: in the chat the word is the student's.
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
  function place(target) {
    // The first line's box when the word wraps: the card belongs to where
    // the word begins.
    const box = target.range.getClientRects()[0] ?? target.range.getBoundingClientRect();
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

  function show(target, parts) {
    card.replaceChildren(...parts);
    card.hidden = false;
    place(target);
    target.owner.setAttribute("aria-describedby", card.id);
  }

  function forget() {
    if (current) current.owner.removeAttribute("aria-describedby");
    current = null;
  }

  function close() {
    clearTimeout(openTimer);
    clearTimeout(closeTimer);
    // Zero, not just cleared: `onMove` reads 0 as "no close pending".
    closeTimer = 0;
    forget();
    card.hidden = true;
  }

  async function open(target) {
    // A dash or a lone quote is a word to the page; there is nothing to look up.
    if (!/[\p{L}\p{N}]/u.test(target.text)) return;
    let parts;
    try {
      parts = render(await lookup(target.text), target.text);
    } catch (error) {
      parts = [el("p", "wc-miss", `The dictionary did not answer: ${error.message}`)];
    }
    // The pointer may have moved on while the lookup ran.
    if (same(current, target)) show(current, parts);
  }

  function leaveSoon() {
    clearTimeout(openTimer);
    if (current) closeTimer = setTimeout(close, CLOSE_DELAY_MS);
  }

  function onMove(event) {
    if (event.pointerType && event.pointerType !== "mouse" && event.pointerType !== "pen") return;
    const target = targetAt(event);
    if (!target) {
      // Off any word — between two, or past the end of a line.
      if (current && !closeTimer) leaveSoon();
      return;
    }
    clearTimeout(closeTimer);
    closeTimer = 0;
    if (same(target, current)) return;
    clearTimeout(openTimer);
    if (current && !card.hidden) {
      // Moving along the line with a card open: follow at once.
      card.hidden = true;
      forget();
      current = target;
      open(target);
      return;
    }
    forget();
    current = target;
    openTimer = setTimeout(() => open(target), OPEN_DELAY_MS);
  }

  for (const root of roots) {
    root.addEventListener("pointermove", onMove);
    root.addEventListener("pointerleave", () => {
      closeTimer = 0;
      leaveSoon();
    });
  }

  card.addEventListener("pointerenter", () => {
    clearTimeout(closeTimer);
    closeTimer = 0;
  });
  card.addEventListener("pointerleave", () => {
    closeTimer = setTimeout(close, CLOSE_DELAY_MS);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !card.hidden) close();
  });
  // Fixed to the viewport, so it would drift off its word on scroll — of the
  // page or of the conversation, which scrolls on its own. A scroll does not
  // bubble, hence the capture — which also hears the card's own scrolling,
  // and reading the senses further down must not close it.
  window.addEventListener(
    "scroll",
    // `instanceof Node`: a scroll can target `window`, which `contains` rejects.
    (event) =>
      card.hidden || (event.target instanceof Node && card.contains(event.target)) || close(),
    { capture: true, passive: true },
  );
  window.addEventListener("resize", () => card.hidden || close());
})();

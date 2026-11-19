"use strict";

// The Phonetics buttons beside a lesson's full text.
//
// The IPA is already in the page, one hidden line under each paragraph, so
// the buttons only rearrange it: no request, nothing to wait for. Loaded only
// when there is IPA to show (the template leaves both out with it off).
//
// Two layouts, switched by the second button once the IPA is showing:
//   - after the paragraph: the line the template rendered, as it is;
//   - between the lines: each written word carries its own IPA under it, so
//     every line of text has its pronunciation right below, wrapping with it.
//     The IPA words come from the paragraph line through the word's
//     `data-ipa` (the indexes lesson.js lights), copied once into
//     `data-ipa-text` for the CSS to draw.

(() => {
  const button = document.querySelector("button.show-ipa");
  const layout = document.querySelector("button.ipa-layout");
  const reading = document.getElementById("full-text-reading");
  if (!button || !reading) return;

  const STORE = "tc.ipaLayout";
  let interlinear = false;
  try { interlinear = localStorage.getItem(STORE) === "interlinear"; } catch { /* no storage */ }

  let annotated = false;
  /** Gives every written word the text of the IPA words it owns. */
  function annotate() {
    if (annotated) return;
    annotated = true;
    for (const sentence of reading.querySelectorAll("p .sentence")) {
      const ipa = reading.querySelector(`.ipa [data-s="${sentence.dataset.s}"]`);
      const used = new Set();
      for (const word of sentence.querySelectorAll(".w")) {
        const tokens = [];
        for (const t of (word.dataset.ipa ?? "").split(" ").filter(Boolean)) {
          // An IPA word shared by two written ones goes under the first.
          if (used.has(t)) continue;
          used.add(t);
          const token = ipa?.querySelector(`.w[data-t="${t}"]`);
          if (token) tokens.push(token.textContent);
        }
        word.dataset.ipaText = tokens.join(" ");
      }
    }
  }

  function render() {
    const show = button.getAttribute("aria-pressed") === "true";
    const between = show && interlinear;
    if (between) annotate();
    reading.classList.toggle("interlinear", between);
    for (const line of reading.querySelectorAll(".ipa")) line.hidden = !show || between;
    if (layout) {
      layout.hidden = !show;
      layout.setAttribute("aria-pressed", String(interlinear));
      layout.textContent = interlinear ? "⇣ After paragraph" : "⇅ Between lines";
      layout.title = interlinear
        ? "Show the IPA after each paragraph instead of under each line"
        : "Show the IPA under each line instead of after each paragraph";
    }
  }

  button.addEventListener("click", () => {
    const show = button.getAttribute("aria-pressed") !== "true";
    button.setAttribute("aria-pressed", String(show));
    button.textContent = show ? "/ə/ Hide phonetics" : "/ə/ Phonetics";
    render();
  });

  layout?.addEventListener("click", () => {
    interlinear = !interlinear;
    try { localStorage.setItem(STORE, interlinear ? "interlinear" : "paragraph"); } catch { /* no storage */ }
    render();
  });
})();

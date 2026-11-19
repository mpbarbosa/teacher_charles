"use strict";

// The Phonetics button beside a lesson's full text.
//
// The IPA is already in the page, one hidden line under each paragraph, so
// the button only toggles it: no request, nothing to wait for. Loaded only
// when there is IPA to show (the template leaves both out with it off).

(() => {
  const button = document.querySelector("button.show-ipa");
  const reading = document.getElementById("full-text-reading");
  if (!button || !reading) return;

  button.addEventListener("click", () => {
    const show = button.getAttribute("aria-pressed") !== "true";
    for (const line of reading.querySelectorAll(".ipa")) line.hidden = !show;
    button.setAttribute("aria-pressed", String(show));
    button.textContent = show ? "/ə/ Hide phonetics" : "/ə/ Phonetics";
  });
})();

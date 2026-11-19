"use strict";

// When each word of a spoken sentence starts, estimated from its letters.
//
// Shared by the two pages that light the word being said: the lesson reader
// (lesson.js) and the phonetics page's Listen (phonetics.js). Loaded before
// either, as a plain script, because the pages have no build step.
//
// ## Lighting a word before its timing arrives
//
// The measured starts come from a Whisper pass (/api/tts/{id}/words) that
// runs after the audio is made, so on a reading started from a click, with
// nothing fetched ahead, they arrived after a short piece had finished playing
// ("ripped from our hands,", 1.5 s of audio: not one word lit) and 3.5 s into
// the next. Until they arrive, the light follows an estimate: the audio's
// length shared out by each word's letters plus one per vowel group, with a
// comma, semicolon, colon or dash worth eight letters of pause, and the last
// 0.1 s left as the tail. Fitted on 14 sentences of a lesson against
// Whisper's starts and checked on the half it was not fitted on: the estimate
// names the word Whisper names at 59-61% of instants, a median of 0.09-0.12 s
// off, where one equal share per word gets 44%.

window.TCWordStarts = (() => {
  const EST_PER_VOWEL_GROUP = 1;
  const EST_CLAUSE_PAUSE = 8;
  const EST_TAIL_SECONDS = 0.1;

  /** When each of `words` starts in `duration` seconds of audio, estimated. */
  function estimate(words, duration) {
    const weights = words.map((word) => {
      const letters = word.replace(/[^\p{L}']/gu, "");
      const vowels = letters.match(/[aeiouy]+/giu)?.length ?? 0;
      return letters.length + EST_PER_VOWEL_GROUP * vowels;
    });
    const pauses = words.map((word, i) =>
      i < words.length - 1 && /[,;:—–]$/u.test(word) ? EST_CLAUSE_PAUSE : 0
    );
    const total = weights.reduce((a, b) => a + b, 0) + pauses.reduce((a, b) => a + b, 0);
    if (!total || !(duration > 0)) return [];
    const span = Math.max(duration - EST_TAIL_SECONDS, duration / 2);
    const starts = [];
    let done = 0;
    weights.forEach((weight, i) => {
      starts.push((span * done) / total);
      done += weight + pauses[i];
    });
    return starts;
  }

  /** The index of the word being said at `t` seconds, or -1 before the first. */
  function at(starts, t) {
    let k = -1;
    while (k + 1 < starts.length && starts[k + 1] <= t) k += 1;
    return k;
  }

  return Object.freeze({ estimate, at });
})();

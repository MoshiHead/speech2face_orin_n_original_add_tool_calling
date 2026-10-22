"""Inject a web answer into PersonaPlex through its native text-conditioning path.

PersonaPlex (moshi-family) has exactly one text input: the agent text stream, fed one
token per LM step with silence forced on the agent audio stream and the sine marker on
the user stream -- the same format as the session-start system prompt
(LMGen._step_text_prompt_core in src/moshi_pkg/moshi/models/lm.py). This module builds
the wrapped text and feeds it mid-session; the model then continues generating speech
conditioned on it.

SAFETY CONTRACT (the server guarantees all of these):
  * run() executes on the persona thread, between two 80 ms audio steps
  * reply_input_lock is held (the LM streaming state cannot be touched mid-step)
  * the same CUDA stream is current that the LM CUDA graphs are replayed on
"""
from __future__ import annotations

import time

from .log import ts_log

WRAP_PREFIX = "<" + "|im_start|>system\n"
WRAP_SUFFIX = "<" + "|im_end|>\n"


def build_injection_text(question: str, answer: str) -> str:
    """Same system wrapper as the production persona prompt (liveTry.py), so the
    model reads the injected text exactly the way it reads its system prompt. The body
    tells it to relay the fact directly -- otherwise it tends to deflect to 'I do not
    have real-time data, check a website'."""
    q = " ".join(str(question or "").split())[:220]
    a = " ".join(str(answer or "").split())
    return (
        WRAP_PREFIX
        + f"Web search result for the user's question \"{q}\": {a}\n"
        + "You now have this current information. Answer the user directly with these "
        + "exact facts, naturally, as the answer -- do not say you lack real-time data "
        + "and do not redirect to a website."
        + WRAP_SUFFIX
    )


class Injection:
    """One pending web-answer injection, executed by the persona thread."""

    def __init__(self, question: str, answer: str, meta: dict | None = None):
        self.question = question
        self.answer = answer
        self.text = build_injection_text(question, answer)
        self.meta = dict(meta or {})

    def fit_token_budget(self, tokenizer, budget: int) -> list:
        """Trim the answer (word boundary) until the wrapped text fits the token budget."""
        tokens = list(tokenizer.encode(self.text))
        words = self.answer.split()
        while len(tokens) > budget and len(words) > 6:
            words = words[: max(6, int(len(words) * 0.8))]
            self.answer = " ".join(words).rstrip(",;") + "."
            self.text = build_injection_text(self.question, self.answer)
            tokens = list(tokenizer.encode(self.text))
        self.meta["tokens"] = len(tokens)
        return tokens

    def run(self, reply_engine, should_abort=None, log=None) -> dict:
        """Feed the tokens into the LM. Never raises; returns a small report dict."""
        log = log or ts_log
        t0 = time.perf_counter()
        fed, total = 0, 0
        try:
            lm_gen = reply_engine.lm_gen
            tokens = self.fit_token_budget(
                reply_engine.tokenizer, int(self.meta.get("budget", 96))
            )
            total = len(tokens)
            lm_gen.text_prompt_tokens = tokens
            try:
                for tok in tokens:
                    if should_abort is not None and should_abort():
                        break
                    lm_gen.step(
                        moshi_tokens=lm_gen._encode_zero_frame(),
                        text_token=tok,
                        input_tokens=lm_gen._encode_sine_frame(),
                    )
                    fed += 1
                if fed and not (should_abort is not None and should_abort()):
                    lm_gen._step_audio_silence()
            finally:
                lm_gen.text_prompt_tokens = None
        except Exception as e:  # noqa: BLE001
            log(f"[INJECT] FAILED ({e!r}) -- continuing without it")
        ms = 1000.0 * (time.perf_counter() - t0)
        aborted = 0 < fed < total
        log(f"[INJECT] fed {fed}/{total} tokens in {ms:.0f} ms"
            f"{' (aborted: barge-in)' if aborted else ''}")
        return {"fed": fed, "total": total, "ms": ms, "aborted": aborted}


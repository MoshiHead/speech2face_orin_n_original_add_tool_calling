"""Deliver a web answer through PersonaPlex by making the model SPEAK it (forced speech).

Why not the text-prompt injection: feeding the answer as a system-wrapped text prompt
(ChatML im_start/im_end system tags) reproduces the exact session-start token sequence,
so PersonaPlex answered with the session-start greeting and ignored the answer
(log/websearch_v4.log). Prompt-wording tweaks could not fix that.

Forced speech instead: each answer token is forced as the agent-TEXT token while the
depformer generates the matching agent-AUDIO (moshi_tokens=None). The model speaks
exactly the answer text in its own voice, through the normal audio path -- no system
prompt, no conversation reset, no greeting. See MoshiOnlyEngineWithHidden._step_forced_text.

SAFETY CONTRACT (the server guarantees all of these):
  * run() executes on the persona thread, between two 80 ms audio steps
  * reply_input_lock is held (the LM streaming state cannot be touched mid-step)
  * the same CUDA stream is current that the LM CUDA graphs are replayed on
"""
from __future__ import annotations

import time

from .log import ts_log


def build_speak_text(question: str, answer: str) -> str:
    """The exact text the model speaks for one question: just the answer, as a natural
    spoken sentence (the answer LLM already writes it that way)."""
    return " ".join(str(answer or "").split())


class Injection:
    """A web answer to be SPOKEN by PersonaPlex (forced agent-text, generated agent-audio).

    Holds the exact text to speak. run() just queues it on the reply engine; the persona
    thread then speaks it token by token through the normal audio path.
    """

    def __init__(self, speak_text: str, meta: dict | None = None):
        self.speak_text = " ".join(str(speak_text or "").split())
        self.text = self.speak_text   # what gets spoken (kept for logging/tests)
        self.meta = dict(meta or {})

    def run(self, reply_engine, should_abort=None, log=None) -> dict:
        """Queue the answer to be spoken by the persona thread. Never raises."""
        log = log or ts_log
        t0 = time.perf_counter()
        try:
            n = reply_engine.queue_forced_speech(self.speak_text)
            ms = 1000.0 * (time.perf_counter() - t0)
            log(f"[INJECT] queued {n} forced-speech tokens: {self.speak_text!r}")
            return {"fed": n, "total": n, "ms": ms, "aborted": False}
        except Exception as e:  # noqa: BLE001
            ms = 1000.0 * (time.perf_counter() - t0)
            log(f"[INJECT] FAILED ({e!r}) -- continuing without it")
            return {"fed": 0, "total": 0, "ms": ms, "aborted": False}

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


def build_injection_text(questions, facts) -> str:
    """Build the mid-session conditioning text (same system wrapper as the production
    persona prompt in liveTry.py, so the model reads it as system text).

    Framing matters: PersonaPlex reads its agent-text stream as its own speech, so an
    injection that states the answer as a finished sentence makes the model believe it
    ALREADY spoke it -- and it then just closes the turn ('let me know if you have any
    other questions') while the user heard silence. So the text always says the user is
    still WAITING and the answer must be spoken OUT LOUD right now.

    Accepts either (question: str, answer: str) or (questions: list, facts: list).
    """
    if isinstance(questions, str):
        questions = [questions]
        facts = [(questions[0], facts)] if isinstance(facts, str) else list(facts or [])
    questions = [" ".join(str(q).split())[:160] for q in questions if str(q).strip()]
    facts = [(" ".join(str(q).split())[:160], " ".join(str(a).split())) for q, a in (facts or [])]
    if len(questions) <= 1:
        q = questions[0] if questions else ""
        a = facts[0][1] if facts else ""
        body = (
            f'The user asked you: "{q}" and is waiting for the answer. '
            f"Tell them now, out loud, using this information: {a} "
            "Answer directly in your own words; do not just offer to help."
        )
    else:
        lines = [
            f"The user asked you {len(questions)} questions and is waiting. "
            "Answer every one of them, out loud, in order:"
        ]
        lines += [f"{i}) {q}" for i, q in enumerate(questions, 1)]
        if facts:
            lines.append("Use this information for the current-information ones:")
            lines += [f'- "{q}": {a}' for q, a in facts]
        lines.append("Now answer all the questions, one by one, in your own words.")
        body = "\n".join(lines)
    return WRAP_PREFIX + body + WRAP_SUFFIX


class Injection:
    """One pending web-answer injection, executed by the persona thread.

    Holds the user's question(s) and the looked-up fact(s) so the wrapped text can be
    rebuilt if it has to be trimmed to the token budget. Accepts either
    (question: str, answer: str) or (questions: list, facts: list[(q, answer)]).
    """

    def __init__(self, questions, facts, meta: dict | None = None):
        if isinstance(questions, str):
            questions = [questions]
            facts = [(questions[0], facts)] if isinstance(facts, str) else list(facts or [])
        self.questions = [str(q) for q in (questions or [])]
        self.facts = [(str(q), str(a)) for q, a in (facts or [])]
        self.question = self.questions[0] if self.questions else ""
        self.answer = self.facts[0][1] if self.facts else ""
        self.text = build_injection_text(self.questions, self.facts)
        self.meta = dict(meta or {})

    def fit_token_budget(self, tokenizer, budget: int) -> list:
        """Trim the longest answer (word boundary) until the wrapped text fits the budget."""
        tokens = list(tokenizer.encode(self.text))
        while len(tokens) > budget and self.facts:
            idx = max(range(len(self.facts)), key=lambda k: len(self.facts[k][1].split()))
            words = self.facts[idx][1].split()
            if len(words) <= 6:
                break
            words = words[: max(6, int(len(words) * 0.8))]
            self.facts[idx] = (self.facts[idx][0], " ".join(words).rstrip(",;") + ".")
            self.text = build_injection_text(self.questions, self.facts)
            tokens = list(tokenizer.encode(self.text))
        self.answer = self.facts[0][1] if self.facts else ""
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
                # NOTE: deliberately NO lm_gen._step_audio_silence() here. At session
                # start the sequence [text prompt][PAD silence] is always immediately
                # followed by the model's greeting, so replaying that exact pattern after
                # a mid-session injection trained the model to greet/close the turn
                # instead of answering. Resuming normal sampling right after the tokens
                # keeps it in "answer mode".
            finally:
                lm_gen.text_prompt_tokens = None
        except Exception as e:  # noqa: BLE001
            log(f"[INJECT] FAILED ({e!r}) -- continuing without it")
        ms = 1000.0 * (time.perf_counter() - t0)
        aborted = 0 < fed < total
        log(f"[INJECT] fed {fed}/{total} tokens in {ms:.0f} ms"
            f"{' (aborted: barge-in)' if aborted else ''}")
        return {"fed": fed, "total": total, "ms": ms, "aborted": aborted}


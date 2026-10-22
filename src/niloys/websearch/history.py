"""Short conversation history (the last N user/assistant turns).

The classifier and the answer synthesizer both receive this so follow-up questions
("and what about its price?") resolve against what was said before. The user side
comes from the STT transcript, the assistant side from PersonaPlex's own text stream
(reply_engine.audio_text deltas between turns).
"""
from __future__ import annotations

import time
from collections import deque


class TurnHistory:
    def __init__(self, max_turns: int = 4):
        self.turns = deque(maxlen=max(1, int(max_turns)))

    def add(self, user_text: str, assistant_text: str) -> None:
        user_text = " ".join(str(user_text or "").split())
        assistant_text = " ".join(str(assistant_text or "").split())
        if not user_text:
            return
        self.turns.append(
            {"user": user_text, "assistant": assistant_text, "ts": time.time()}
        )

    def as_prompt_text(self, max_chars: int = 900) -> str:
        if not self.turns:
            return ""
        lines = []
        for t in self.turns:
            a = t["assistant"]
            if len(a) > 220:
                a = a[:220].rsplit(" ", 1)[0] + "..."
            lines.append(f"User: {t['user']}\nAssistant: {a or '(no reply)'}")
        return "\n".join(lines)[-max_chars:]

    def clear(self) -> None:
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)

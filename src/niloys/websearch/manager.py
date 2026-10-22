"""WebSearchManager -- the facade the live server talks to.

Owns the background model load (STT + LLM), the per-session orchestrator and every
thread boundary. All methods are safe to call before the models finish loading and
never raise into the server: the classic pipeline must survive any web-search failure.
"""
from __future__ import annotations

import threading
import time

from .log import ts_log


class WebSearchManager:
    def __init__(self, config, log=None):
        self.cfg = config
        self.log = log or getattr(config, "log", None) or ts_log
        self.ready = False
        self.error: str | None = None
        self.stt = None
        self.llm = None
        self.orchestrator = None
        self._load_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ load
    def start_background_load(self) -> None:
        self._load_thread = threading.Thread(
            target=self._load, name="websearch-load", daemon=True
        )
        self._load_thread.start()

    def _load(self) -> None:
        try:
            from .llm_backend import LocalLLM
            from .orchestrator import WebSearchOrchestrator
            from .stt_engine import KyutaiSTT

            t0 = time.perf_counter()
            self.llm = LocalLLM(self.cfg.llm_model, self.cfg.device, log=self.log)
            self.log(f"[LLM] ready: {self.cfg.llm_model}")
            self.stt = KyutaiSTT(
                self.cfg.stt_repo,
                self.cfg.device,
                int4=self.cfg.stt_int4,
                vad_rms=self.cfg.vad_rms,
                vad_speech_ms=self.cfg.vad_speech_ms,
                vad_silence_ms=self.cfg.vad_silence_ms,
                flush_quiet_ms=self.cfg.flush_quiet_ms,
                flush_max_ms=self.cfg.flush_max_ms,
                min_question_words=self.cfg.min_question_words,
                assistant_rms=self.cfg.assistant_rms,
                assistant_hold_s=self.cfg.assistant_hold_s,
                log=self.log,
            )
            self.log(f"[STT] ready: {self.cfg.stt_repo}")
            self.orchestrator = WebSearchOrchestrator(self.cfg, self.stt, self.llm, log=self.log)
            self.stt.on_turn_complete = self.orchestrator.on_turn_complete
            self.stt.start()
            # Warm up the STT on a short silence buffer: any lazy cuDNN/autotune work
            # happens here in the background thread (not mid-conversation), and it
            # validates the model before it goes live.
            try:
                import numpy as np
                self.stt.transcribe_offline(np.zeros(24000, dtype=np.float32))
                self.log("[STT] warmup ok")
            except Exception as e:  # noqa: BLE001
                self.log(f"[STT] warmup failed ({e!r}) -- will retry on the first turn")
            self.ready = True
            self.log(
                f"ready in {time.perf_counter() - t0:.1f}s "
                f"(device={self.cfg.device}, dry_run={self.cfg.dry_run}, "
                f"history={self.cfg.history_turns} turns)"
            )
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self.log(f"DISABLED -- model load failed: {self.error}")

    # ------------------------------------------------------------------ server hooks
    def feed_pcm24(self, pcm) -> None:
        if self.ready and self.stt is not None:
            self.stt.feed(pcm)

    def observe_reply_rms(self, rms: float) -> None:
        if self.ready and self.orchestrator is not None:
            self.orchestrator.observe_reply_rms(rms)

    def on_session_start(self, loop, get_reply_text) -> None:
        if self.ready and self.orchestrator is not None:
            self.orchestrator.on_session_start(loop, get_reply_text)

    def on_session_end(self) -> None:
        if self.ready and self.orchestrator is not None:
            self.orchestrator.on_session_end()

    def pop_injection(self):
        if self.ready and self.orchestrator is not None:
            return self.orchestrator.pop_injection()
        return None

    def should_abort_injection(self) -> bool:
        if self.ready and self.orchestrator is not None:
            return self.orchestrator.should_abort_injection()
        return True

    def status(self) -> dict:
        if self.error:
            return {"enabled": True, "ready": False, "error": self.error}
        return {
            "enabled": True,
            "ready": self.ready,
            "llm": self.cfg.llm_model,
            "stt": self.cfg.stt_repo,
            "dry_run": self.cfg.dry_run,
        }

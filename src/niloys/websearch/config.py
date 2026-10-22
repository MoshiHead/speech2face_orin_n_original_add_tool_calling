"""Configuration for the web-search augmentation.

Pipeline: kyutai/stt-1b-en_fr transcribes the mic -> a small local instruct LLM
(downloaded from Hugging Face, no API) decides whether the question needs current
online information -> Tavily top-3 search -> the same LLM writes a short answer ->
the answer is fed into PersonaPlex through its native text-prompt stream.

Everything is driven by CLI args (see LiveHeliumFMOptions) with env-var fallbacks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from .log import ts_log


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "no", "false", "")


@dataclass
class WebSearchConfig:
    tavily_api_key: str
    stt_repo: str = "kyutai/stt-1b-en_fr"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cuda"
    stt_int4: bool = False
    history_turns: int = 4
    max_inject_tokens: int = 96
    max_answer_words: int = 60
    dry_run: bool = False
    vad_rms: float = 0.01
    vad_speech_ms: int = 500
    vad_silence_ms: int = 500
    flush_quiet_ms: int = 500
    flush_max_ms: int = 1400
    min_question_words: int = 2
    inject_deadline_ms: int = 3500
    search_timeout_s: float = 6.0
    llm_timeout_s: float = 8.0
    tavily_include_answer: bool = False
    max_results: int = 3
    assistant_rms: float = 0.006
    assistant_hold_s: float = 0.5
    log: Optional[Callable[..., None]] = None

    def __post_init__(self) -> None:
        if self.log is None:
            self.log = ts_log

    @classmethod
    def from_args(cls, args, log: Optional[Callable[..., None]] = None) -> Optional["WebSearchConfig"]:
        """Build the config from parsed server args, or None when the feature must stay off.

        The feature is requested with --enable_web_search but only activates with a
        Tavily API key; without one the server runs the classic pipeline unchanged.
        """
        if not bool(getattr(args, "enable_web_search", False)):
            return None
        log = log or ts_log
        key = str(
            getattr(args, "tavily_api_key", "") or os.environ.get("TAVILY_API_KEY", "")
        ).strip()
        if not key:
            log(
                "--enable_web_search given but no Tavily API key "
                "(set TAVILY_API_KEY or --tavily_api_key) -- running the classic pipeline"
            )
            return None
        device = str(getattr(args, "websearch_device", "") or getattr(args, "device", "cuda"))
        return cls(
            tavily_api_key=key,
            stt_repo=str(getattr(args, "websearch_stt_repo", cls.stt_repo)),
            llm_model=str(getattr(args, "websearch_llm_model", cls.llm_model)),
            device=device,
            stt_int4=bool(getattr(args, "websearch_stt_int4", False)) or _env_flag("WEBSEARCH_STT_INT4"),
            history_turns=int(getattr(args, "websearch_history_turns", 4)),
            max_inject_tokens=int(getattr(args, "websearch_max_inject_tokens", 96)),
            dry_run=bool(getattr(args, "websearch_dry_run", False)) or _env_flag("WEBSEARCH_DRY_RUN"),
            vad_silence_ms=int(getattr(args, "websearch_vad_silence_ms", 500)),
            inject_deadline_ms=int(getattr(args, "websearch_inject_deadline_ms", 3500)),
            tavily_include_answer=bool(getattr(args, "websearch_tavily_include_answer", False))
            or _env_flag("WEBSEARCH_TAVILY_INCLUDE_ANSWER"),
            log=log,
        )

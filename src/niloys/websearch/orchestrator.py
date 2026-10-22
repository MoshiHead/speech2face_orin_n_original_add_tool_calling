"""Turn pipeline: STT turn -> classify -> Tavily -> synthesize -> queue the injection.

Runs on the server's asyncio loop (search is aiohttp; the two LLM calls go to the
default executor because they are blocking GPU work). Everything is best-effort: any
failure or timeout simply skips the injection and the classic pipeline answers.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time

from .history import TurnHistory
from .injector import Injection
from .llm_backend import keyword_suggests_search, needs_context
from .tavily_client import format_results, tavily_search_async


class WebSearchOrchestrator:
    def __init__(self, config, stt, llm, log=print):
        self.cfg = config
        self.stt = stt
        self.llm = llm
        self.log = log
        self.history = TurnHistory(config.history_turns)
        self._injections: queue.Queue = queue.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._get_reply_text = None
        self._pending_task: asyncio.Task | None = None
        self._reply_task: asyncio.Task | None = None
        self._pending_user_q: str | None = None
        self._reply_snapshot = ""
        self._session_stop = threading.Event()

    # ------------------------------------------------------- session lifecycle
    def on_session_start(self, loop, get_reply_text) -> None:
        self._loop = loop
        self._get_reply_text = get_reply_text
        self._session_stop.clear()
        self.history.clear()
        self._pending_user_q = None
        self._reply_snapshot = ""
        self._cancel_pending()
        self._drain_injections()
        self.stt.session_start()

    def on_session_end(self) -> None:
        self._session_stop.set()
        self._cancel_pending()
        self._drain_injections()
        self.stt.session_end()

    # ------------------------------------------------------- callbacks
    def on_turn_complete(self, question: str, turn_end_time: float) -> None:
        """Called from the STT thread when a user question is fully transcribed."""
        loop = self._loop
        if loop is None or self._session_stop.is_set():
            return
        loop.call_soon_threadsafe(self._schedule_turn, question, turn_end_time)

    def observe_reply_rms(self, rms: float) -> None:
        self.stt.note_assistant_rms(rms)

    # ------------------------------------------------------- persona thread
    def pop_injection(self):
        try:
            return self._injections.get_nowait()
        except queue.Empty:
            return None

    def should_abort_injection(self) -> bool:
        return self._session_stop.is_set() or self.stt.user_speaking_recently(0.25)

    # ------------------------------------------------------- internals (event loop)
    def _schedule_turn(self, question: str, turn_end_time: float) -> None:
        if self._session_stop.is_set():
            return
        self._close_previous_turn()
        self._pending_user_q = question
        self._cancel_pending()
        self._pending_task = self._loop.create_task(self._pipeline(question, turn_end_time))

    def _close_previous_turn(self) -> None:
        """Store (previous question, assistant reply delta) into the history."""
        try:
            now_text = str(self._get_reply_text() or "") if self._get_reply_text else ""
            if self._pending_user_q is not None:
                delta = now_text[len(self._reply_snapshot):] if len(now_text) >= len(self._reply_snapshot) else ""
                self.history.add(self._pending_user_q, delta)
            self._reply_snapshot = now_text
        except Exception:  # noqa: BLE001
            pass
        self._pending_user_q = None

    def _cancel_pending(self) -> None:
        if self._pending_task is not None and not self._pending_task.done():
            self._pending_task.cancel()
        self._pending_task = None
        if self._reply_task is not None and not self._reply_task.done():
            self._reply_task.cancel()
        self._reply_task = None

    def _drain_injections(self) -> None:
        while True:
            try:
                self._injections.get_nowait()
            except queue.Empty:
                break

    async def _pipeline(self, question: str, turn_end_time: float) -> None:
        cfg = self.cfg
        t0 = time.perf_counter()
        try:
            history_text = self.history.as_prompt_text()
            if keyword_suggests_search(question) and not needs_context(question):
                # keyword fast-path: an obvious, self-contained current-info question
                # goes straight to search (a conservative small LLM cannot suppress it)
                needs, query = True, question
                self.log(f"[LLM-DECISION] SEARCH (keyword fast-path) q={question!r}")
            else:
                needs, query = await asyncio.wait_for(
                    self._loop.run_in_executor(None, self.llm.classify, question, history_text),
                    timeout=cfg.llm_timeout_s,
                )
                raw = getattr(self.llm, "last_classify_raw", "")
                self.log(f"[LLM-DECISION] {'SEARCH' if needs else 'NOSEARCH'}"
                         f" ({1000.0 * (time.perf_counter() - t0):.0f} ms)"
                         f" q={question!r} raw={raw!r}")
            if not needs:
                return
            t1 = time.perf_counter()
            data = await asyncio.wait_for(
                tavily_search_async(query, cfg.tavily_api_key, cfg.max_results,
                                    cfg.search_timeout_s, cfg.tavily_include_answer),
                timeout=cfg.search_timeout_s + 1.0,
            )
            results = format_results(data, cfg.max_results)
            self.log(f"[SEARCH] {len(results)} results"
                     f" ({1000.0 * (time.perf_counter() - t1):.0f} ms) query={query!r}")
            for i, r in enumerate(results, 1):
                self.log(f"[SEARCH]   [{i}] {r.get('title', '')} ({r.get('url', '')})")
                self.log(f"[SEARCH]       {r.get('content', '')[:180]}")
            if not results:
                return
            t2 = time.perf_counter()
            answer = await asyncio.wait_for(
                self._loop.run_in_executor(None, self.llm.synthesize, question, results,
                                           history_text, cfg.max_answer_words),
                timeout=cfg.llm_timeout_s,
            )
            if not answer:
                # one retry with a bare prompt (no history) before giving up
                answer = await asyncio.wait_for(
                    self._loop.run_in_executor(None, self.llm.synthesize, question, results,
                                               "", cfg.max_answer_words),
                    timeout=cfg.llm_timeout_s,
                )
            self.log(f"[LLM-SUMMARY] ({1000.0 * (time.perf_counter() - t2):.0f} ms)"
                     f" answer: {answer!r}")
            if not answer:
                return
            late_ms = 1000.0 * (time.perf_counter() - turn_end_time)
            if late_ms > cfg.inject_deadline_ms:
                self.log(f"[INJECT] answer ready {late_ms:.0f} ms after turn end"
                         f" (> {cfg.inject_deadline_ms} ms deadline) -- NOT injecting")
                return
            inj = Injection(question, answer, meta={
                "budget": cfg.max_inject_tokens,
                "query": query,
                "pipeline_ms": 1000.0 * (time.perf_counter() - t0),
            })
            if cfg.dry_run:
                self.log(f"[INJECT] DRY RUN -- would inject: {inj.text!r}")
            else:
                self.log(f"[INJECT] queued: {inj.text!r}")
                self._injections.put(inj)
                self._schedule_reply_log()
        except asyncio.CancelledError:
            self.log("[SEARCH] superseded by a newer question")
            raise
        except Exception as e:  # noqa: BLE001
            self.log(f"[SEARCH] pipeline failed ({e!r}) -- continuing without search")

    # ------------------------------------------------------- reply capture
    def _schedule_reply_log(self) -> None:
        """Log what the model actually says after the injection lands."""
        if self._get_reply_text is None or self._loop is None:
            return
        snapshot = str(self._get_reply_text() or "")
        if self._reply_task is not None and not self._reply_task.done():
            self._reply_task.cancel()
        self._reply_task = self._loop.create_task(self._reply_logger(snapshot))

    async def _reply_logger(self, snapshot: str) -> None:
        try:
            await asyncio.sleep(2.5)   # let the injection land and the model start speaking
            last, stable = snapshot, 0
            for _ in range(30):        # up to ~30 s of reply
                await asyncio.sleep(1.0)
                now = str(self._get_reply_text() or "")
                if not now.startswith(snapshot):
                    return             # session was reset mid-reply
                if len(now) == len(last):
                    stable += 1
                    if stable >= 2:
                        break
                else:
                    stable = 0
                    last = now
            delta = " ".join(last[len(snapshot):].split())
            if delta:
                self.log(f"[REPLY] model said: {delta!r}")
            else:
                self.log("[REPLY] (no spoken reply captured)")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self.log(f"[REPLY] capture failed ({e!r})")


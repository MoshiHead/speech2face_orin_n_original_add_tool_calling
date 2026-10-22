"""Timestamped logger for the web-search pipeline.

Every stage prints on its own tag so run.log reads like a trace of the turn:

    [WEBSEARCH][12:34:56.789] [STT] question: 'what is the bitcoin price today?'
    [WEBSEARCH][12:34:57.012] [LLM-DECISION] SEARCH (234 ms) q='...' raw='SEARCH\nbitcoin price'
    [WEBSEARCH][12:34:58.345] [SEARCH] 3 results (1102 ms) query='bitcoin price today'
    [WEBSEARCH][12:34:58.345] [SEARCH]   [1] Bitcoin price today (https://...)
    [WEBSEARCH][12:34:58.345] [SEARCH]       Bitcoin is trading at about ...
    [WEBSEARCH][12:34:59.678] [LLM-SUMMARY] (1333 ms) answer: 'Bitcoin is ...'
    [WEBSEARCH][12:35:00.901] [INJECT] queued: '<system> Web search result ...'
    [WEBSEARCH][12:35:01.234] [INJECT] fed 72/72 tokens in 890 ms
    [WEBSEARCH][12:35:08.456] [REPLY] model said: 'Sure - bitcoin is currently ...'
"""
from __future__ import annotations

import time


def ts_log(msg: str) -> None:
    """Print one timestamped web-search log line (millisecond resolution, flushed)."""
    t = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
    print(f"[WEBSEARCH][{t}] {msg}", flush=True)
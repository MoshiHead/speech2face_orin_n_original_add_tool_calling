"""Timestamped logger for the web-search pipeline.

Every stage prints on its own tag so the log reads like a trace of the turn:

    [WEBSEARCH][12:34:56.789] [STT] question: 'what is the bitcoin price today?'
    [WEBSEARCH][12:34:57.012] [LLM-DECISION] SEARCH (234 ms) q='...' raw='SEARCH\nbitcoin price'
    [WEBSEARCH][12:34:58.345] [SEARCH] 3 results (1102 ms) query='bitcoin price today'
    [WEBSEARCH][12:34:58.345] [SEARCH]   [1] Bitcoin price today (https://...)
    [WEBSEARCH][12:34:58.345] [SEARCH]       Bitcoin is trading at about ...
    [WEBSEARCH][12:34:59.678] [LLM-SUMMARY] (1333 ms) answer: 'Bitcoin is ...'
    [WEBSEARCH][12:35:00.901] [INJECT] queued: '<system> Web search result ...'
    [WEBSEARCH][12:35:01.234] [INJECT] fed 72/72 tokens in 890 ms
    [WEBSEARCH][12:35:08.456] [REPLY] model said: 'Sure - bitcoin is currently ...'

Every line goes to stdout (run.log) AND to a dedicated web-search log file so the
web-search trace is easy to read on its own:
  * file path:  env WEBSEARCH_LOG_FILE  (default "websearch.log" in the repo cwd)
  * stdout copy: env WEBSEARCH_LOG_STDOUT=0 turns it off (file only)
"""
from __future__ import annotations

import os
import time

_FILE_HANDLE = None
_FILE_FAILED = False


def _log_path() -> str:
    return os.environ.get("WEBSEARCH_LOG_FILE", "websearch.log")


def _stdout_enabled() -> bool:
    return os.environ.get("WEBSEARCH_LOG_STDOUT", "1").strip().lower() not in ("0", "no", "false")


def _get_file():
    global _FILE_HANDLE, _FILE_FAILED
    if _FILE_HANDLE is None and not _FILE_FAILED:
        try:
            # line-buffered so a tail -f sees lines as they happen
            _FILE_HANDLE = open(_log_path(), "a", buffering=1, encoding="utf-8")
        except Exception:
            _FILE_FAILED = True
    return _FILE_HANDLE


def ts_log(msg: str) -> None:
    """Print one timestamped web-search log line to stdout and the web-search log file."""
    t = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
    line = f"[WEBSEARCH][{t}] {msg}"
    if _stdout_enabled():
        print(line, flush=True)
    fh = _get_file()
    if fh is not None:
        try:
            fh.write(line + "\n")
        except Exception:
            pass

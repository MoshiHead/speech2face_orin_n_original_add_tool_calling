"""tools/test_websearch_units.py -- pure-python unit tests for the web-search package.

No GPU, no models, no network: tests the parsing/formatting/history/injection-text
logic that does not need torch. Run:  python tools/test_websearch_units.py
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "niloys"))

from websearch.history import TurnHistory                      # noqa: E402
from websearch.injector import Injection, build_speak_text     # noqa: E402
from websearch.llm_backend import (parse_classification, is_stable_question, split_questions,  # noqa: E402
                                   keyword_suggests_search, needs_context)  # noqa: E402
from websearch.tavily_client import format_results              # noqa: E402
from websearch.config import WebSearchConfig                    # noqa: E402
from websearch.stt_engine import KyutaiSTT, FRAME_SAMPLES       # noqa: E402
import numpy as np                                              # noqa: E402
import queue as _queue                                          # noqa: E402
import threading as _threading                                  # noqa: E402
import time as _time                                            # noqa: E402
from collections import deque as _deque                         # noqa: E402

FAILED = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------- classification parsing
check("classify SEARCH with query",
      parse_classification("SEARCH\nbitcoin price today", "q?") == (True, "bitcoin price today"))
check("classify NOSEARCH", parse_classification("NOSEARCH", "q?") == (False, "q?"))
check("classify bare SEARCH falls back to question",
      parse_classification("SEARCH", "the question?") == (True, "the question?"))
check("classify 'no search' is NOSEARCH", parse_classification("No search needed", "q?") == (False, "q?"))
check("classify empty is NOSEARCH", parse_classification("", "q?") == (False, "q?"))
check("classify garbage is NOSEARCH", parse_classification("I'm not sure", "q?") == (False, "q?"))
check("classify NOSEARCH beats SEARCH substring",
      parse_classification("NOSEARCH\nSEARCH for nothing", "q?") == (False, "q?"))

# ---------------------------------------------------------------- tavily formatting
fake = {"results": [
    {"title": f"  title {i}  ", "content": "word " * 400, "url": f"http://x/{i}", "score": 0.9}
    for i in range(5)
]}
res = format_results(fake, 3)
check("tavily keeps top-3", len(res) == 3)
check("tavily trims content", all(len(r["content"]) <= 700 for r in res))
check("tavily strips title whitespace", res[0]["title"] == "title 0")
check("tavily empty response", format_results({}, 3) == [])
check("tavily drops empty content",
      format_results({"results": [{"title": "t", "content": "  "}, {"title": "u", "content": "real"}]}, 3)
      == [{"title": "u", "content": "real", "url": ""}])

# ---------------------------------------------------------------- history
h = TurnHistory(4)
for i in range(6):
    h.add(f"question {i}?", f"answer {i}")
check("history keeps last 4", len(h) == 4)
text = h.as_prompt_text()
check("history text has user+assistant", "User: question 5?" in text and "Assistant: answer 5" in text)
check("history dropped oldest", "question 0" not in text and "question 1" not in text)
h.add("  spaced   out  ", "")
check("history normalizes whitespace", h.turns[-1]["user"] == "spaced out")
h.clear()
check("history clear", len(h) == 0 and h.as_prompt_text() == "")

# ---------------------------------------------------------------- forced-speech injection
check("speak text is the answer", build_speak_text("q?", "The price is $100.") == "The price is $100.")
check("speak text normalizes whitespace", build_speak_text("q?", "  spaced   out  ") == "spaced out")


class FakeEngine:
    def __init__(self):
        self.queued = []

    def queue_forced_speech(self, text):
        self.queued.append(text)
        return len(text.split())


inj = Injection("The current price of gold is $4,214.85 per ounce.")
eng = FakeEngine()
report = inj.run(eng, log=lambda m: None)
check("injection queues the speak text", eng.queued == ["The current price of gold is $4,214.85 per ounce."])
check("injection reports fed tokens", report["fed"] == 9 and report["total"] == 9 and not report["aborted"])
check("injection text is the speak text", inj.text == "The current price of gold is $4,214.85 per ounce.")


class BadEngine:
    def queue_forced_speech(self, text):
        raise RuntimeError("boom")


report2 = Injection("some answer").run(BadEngine(), log=lambda m: None)
check("injection survives engine failure", report2["fed"] == 0)

# ---------------------------------------------------------------- stable-question guard
check("stable: what is X", is_stable_question("What is Bitcoin?"))
check("stable: explain", is_stable_question("Explain how a blockchain works."))
check("stable: opinion", is_stable_question("Is it safe to invest in Bitcoin?"))
check("stable: how does", is_stable_question("How does bitcoin mining work?"))
check("stable: who is", is_stable_question("Who is Satoshi Nakamoto?"))
check("not stable: price", not is_stable_question("What is the price of Bitcoin?"))
check("not stable: today", not is_stable_question("What is the weather today?"))
check("not stable: news", not is_stable_question("What is the latest news?"))
check("not stable: current", not is_stable_question("What is the current gold rate?"))
check("not stable: empty", not is_stable_question(""))

# ---------------------------------------------------------------- split questions
check("split single", split_questions("What is Bitcoin?") == ["What is Bitcoin?"])
check("split no question mark", split_questions("tell me about bitcoin") == ["tell me about bitcoin"])
check("split four", split_questions("What is Bitcoin? What is its price? Is it safe? What is gold today?")
      == ["What is Bitcoin?", "What is its price?", "Is it safe?", "What is gold today?"])
check("split empty", split_questions("") == [])


# ---------------------------------------------------------------- end-to-end routing decision
def _route(q):
    """Mirrors _classify_one's deterministic path: keyword fast-path -> stable guard.
    Returns True=SEARCH, False=NOSEARCH, None=would fall through to the LLM."""
    if keyword_suggests_search(q) and not needs_context(q):
        return True
    if is_stable_question(q):
        return False
    return None


check("route: what is bitcoin -> NOSEARCH", _route("What is Bitcoin?") is False)
check("route: current price of bitcoin -> SEARCH", _route("What is the current price of Bitcoin?") is True)
check("route: is it safe to invest -> NOSEARCH", _route("Is it safe to invest in Bitcoin?") is False)
check("route: gold price today -> SEARCH", _route("What is the gold price today?") is True)
check("route: how does mining work -> NOSEARCH", _route("How does bitcoin mining work?") is False)
check("route: explain blockchain -> NOSEARCH", _route("Explain how a blockchain works.") is False)
check("route: latest news -> SEARCH", _route("What is the latest news?") is True)

# ---------------------------------------------------------------- config from args
def _args(**kw):
    base = dict(enable_web_search=False, tavily_api_key="", device="cuda",
                websearch_stt_repo="kyutai/stt-1b-en_fr",
                websearch_llm_model="Qwen/Qwen2.5-1.5B-Instruct",
                websearch_device="", websearch_stt_int4=False,
                websearch_history_turns=4, websearch_max_inject_tokens=96,
                websearch_dry_run=False, websearch_vad_silence_ms=500,
                websearch_inject_deadline_ms=3500, websearch_tavily_include_answer=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


logs = []
check("config off without flag", WebSearchConfig.from_args(_args(), log=logs.append) is None)
check("config off without key",
      WebSearchConfig.from_args(_args(enable_web_search=True), log=logs.append) is None)
cfg = WebSearchConfig.from_args(_args(enable_web_search=True, tavily_api_key="tvly-test"), log=logs.append)
check("config on with key", cfg is not None and cfg.tavily_api_key == "tvly-test")
check("config defaults", cfg.llm_model == "Qwen/Qwen2.5-1.5B-Instruct"
      and cfg.history_turns == 4 and cfg.max_inject_tokens == 96 and not cfg.dry_run)
os.environ["TAVILY_API_KEY"] = "tvly-env"
cfg2 = WebSearchConfig.from_args(_args(enable_web_search=True), log=logs.append)
check("config key from env", cfg2 is not None and cfg2.tavily_api_key == "tvly-env")
del os.environ["TAVILY_API_KEY"]


# ---------------------------------------------------------------- STT turn logic (stubbed model)
# KyutaiSTT.__init__ loads the real checkpoint, so build a bare instance and stub
# the model/reset: this exercises the VAD, the pre-roll replay, the echo guard and
# the empty-turn diagnostics without a GPU.
def _make_stub_stt():
    stt = object.__new__(KyutaiSTT)
    stt.logs = []
    stt.log = lambda msg: stt.logs.append(str(msg))
    stt.vad_rms = 0.01
    stt.vad_speech_ms = 500
    stt.vad_silence_ms = 500
    stt.flush_quiet_ms = 500
    stt.flush_max_ms = 1400
    stt.min_question_words = 2
    stt.assistant_rms = 0.006
    stt.assistant_hold_s = 0.5
    stt.preroll_frames = 10
    stt._q = _queue.Queue(maxsize=600)
    stt._stop = _threading.Event()
    stt._lock = _threading.Lock()
    stt._session_active = True
    stt.on_turn_complete = None
    stt.on_turn_closing = None
    stt._preroll = _deque(maxlen=stt.preroll_frames)
    stt._reset_model_state = lambda: None
    stt._model_calls = []
    def _fake_model_frame(frame):
        stt._model_calls.append(float(np.sqrt(np.mean(np.square(frame)))))
        return ""
    stt._model_frame = _fake_model_frame
    KyutaiSTT._reset_turn_state(stt)
    return stt


def _frame(rms):
    return np.full(FRAME_SAMPLES, rms, dtype=np.float32)


stt = _make_stub_stt()
for _ in range(3):
    stt._process_frame(_frame(0.005))          # soft onset frames -> pre-roll
check("preroll buffers pre-turn frames", len(stt._preroll) == 3)
stt._process_frame(_frame(0.05))               # loud frame -> turn opens
check("turn opens on loud frame", stt._in_turn is True)
check("preroll replayed into model at turn open", len(stt._model_calls) == 4)  # 3 pre + 1 live
check("preroll cleared after turn open", len(stt._preroll) == 0)

stt = _make_stub_stt()
stt._assistant_speaking_until = _time.perf_counter() + 10.0   # avatar speaking
for _ in range(3):
    stt._process_frame(_frame(0.005))
check("echo guard keeps pre-roll empty", len(stt._preroll) == 0)
check("echo guard blocks turn open", stt._in_turn is False)

stt = _make_stub_stt()
stt._process_frame(_frame(0.05))
for _ in range(8):
    stt._process_frame(_frame(0.05))           # speech, but the model emits nothing
stt._fire_turn(_time.perf_counter())
check("empty turn logs DIAG", any("DIAG empty turn" in m for m in stt.logs))

stt = _make_stub_stt()
_pieces = iter([" what", " is", " bitcoin", "?", "", "", "", "", "", "", "", ""])
stt._model_frame = lambda frame: next(_pieces, "")
stt._process_frame(_frame(0.05))
for _ in range(10):
    stt._process_frame(_frame(0.05))
_got = []
stt.on_turn_complete = lambda q, t: _got.append(q)
stt._fire_turn(_time.perf_counter())
check("turn assembles question text", bool(_got) and "bitcoin" in _got[0])

print()
if FAILED:
    print(f"{len(FAILED)} FAILURES: {FAILED}")
    sys.exit(1)
print("ALL UNIT TESTS PASS")

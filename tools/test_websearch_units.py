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
from websearch.injector import Injection, build_injection_text  # noqa: E402
from websearch.llm_backend import parse_classification          # noqa: E402
from websearch.tavily_client import format_results              # noqa: E402
from websearch.config import WebSearchConfig                    # noqa: E402

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

# ---------------------------------------------------------------- injection text
class FakeTok:
    def encode(self, text):
        return text.split()


t = build_injection_text("what is the bitcoin price?", "It is about $100,000 today.")
check("injection wraps in system tags", t.startswith("<" + "|im_start|>system\n") and t.endswith("<" + "|im_end|>\n"))
check("injection contains question and answer", "bitcoin price?" in t and "$100,000 today" in t)

inj = Injection("q?", " ".join(f"w{i}" for i in range(40)), meta={"budget": 30})
tokens = inj.fit_token_budget(FakeTok(), 30)
check("injection fits token budget", len(tokens) <= 30)
check("injection keeps at least 6 answer words", len(inj.answer.split()) >= 6)
check("injection records token count", inj.meta["tokens"] == len(tokens))

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

print()
if FAILED:
    print(f"{len(FAILED)} FAILURES: {FAILED}")
    sys.exit(1)
print("ALL UNIT TESTS PASS")

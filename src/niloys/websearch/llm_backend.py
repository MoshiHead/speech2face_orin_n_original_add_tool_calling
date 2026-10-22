"""Local small instruct LLM (Hugging Face download, NO API) for search routing + answers.

Default model: Qwen/Qwen2.5-1.5B-Instruct (bf16, ~3.1 GB, ungated, Apache-2.0) -- one of
the strongest sub-2B instruct models, and small enough to sit next to the int4
PersonaPlex trunk on a 24 GB card. It needs transformers >= 4.37 (qwen2 arch); the x86
env installs 4.37.2 via requirements-x86-extra.txt. Any other HF causal-LM repo id can
be passed instead (e.g. Qwen/Qwen2.5-0.5B-Instruct for tighter VRAM).

Two jobs, both greedy and short:
  1. classify(question, history)  -> (needs_search, search_query)
  2. synthesize(question, results, history) -> one concise spoken-style answer
"""
from __future__ import annotations

import threading

CLASSIFY_SYSTEM = (
    "You route spoken questions. Reply SEARCH only if answering needs current, recent or "
    "external information: news, prices, weather, sports results, releases, events, "
    "company or product facts, or specific facts a small assistant may not know. "
    "Reply NOSEARCH for casual chat, opinions, math, explanations of well-known concepts, "
    "or questions about the assistant.\n"
    "Answer with EXACTLY one line: SEARCH or NOSEARCH. "
    "If SEARCH, add a second line with a short web search query."
)

ANSWER_SYSTEM = (
    "You answer the user's question using ONLY the web results provided. "
    "Be concise: at most {max_words} words, plain spoken sentences, no lists, no markdown, "
    "no URLs. If the results do not answer the question, say you could not find it."
)


class LocalLLM:
    def __init__(self, model_name: str, device: str = "cuda", log=print):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.log = log
        self.device = torch.device(device)
        self._lock = threading.Lock()
        self.last_classify_raw = ""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, messages: list, max_new_tokens: int = 96) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        with self._lock, torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def classify(self, question: str, history_text: str = "") -> tuple:
        """-> (needs_search: bool, search_query: str). Any failure -> NOSEARCH."""
        user = ""
        if history_text:
            user += f"Conversation so far:\n{history_text}\n\n"
        user += f"User's question: {question}"
        try:
            text = self.generate(
                [
                    {"role": "system", "content": CLASSIFY_SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_new_tokens=24,
            )
        except Exception as e:  # noqa: BLE001
            self.log(f"[LLM] classifier failed ({e!r}) -> NOSEARCH")
            self.last_classify_raw = ""
            return False, question
        self.last_classify_raw = text
        return parse_classification(text, question)

    def synthesize(self, question: str, results: list, history_text: str = "", max_words: int = 60) -> str:
        blocks = []
        for i, r in enumerate(results, 1):
            blocks.append(f"[{i}] {r.get('title', '')}\n{r.get('content', '')}")
        user = ""
        if history_text:
            user += f"Conversation so far:\n{history_text}\n\n"
        user += f"Question: {question}\n\nWeb results:\n" + "\n\n".join(blocks)
        try:
            return self.generate(
                [
                    {"role": "system", "content": ANSWER_SYSTEM.format(max_words=max_words)},
                    {"role": "user", "content": user},
                ],
                max_new_tokens=140,
            )
        except Exception as e:  # noqa: BLE001
            self.log(f"[LLM] answer synthesis failed ({e!r})")
            return ""


def parse_classification(text: str, question: str) -> tuple:
    """Robustly parse the classifier output. Default: NOSEARCH (preserve the pipeline)."""
    lines = [l.strip() for l in str(text or "").splitlines() if l.strip()]
    if not lines:
        return False, question
    first = lines[0].upper()
    if "NOSEARCH" in first or "NO SEARCH" in first:
        return False, question
    if "SEARCH" in first:
        query = lines[1] if len(lines) > 1 else question
        query = query.strip().strip('"').strip() or question
        return True, query
    return False, question


# Obvious "needs current information" signals. A self-contained question with one of
# these is sent to Tavily directly (the keyword fast-path), so a conservative small LLM
# can never suppress an obvious search. Ambiguous / follow-up questions still go to the
# LLM classifier (which sees the conversation history).
SEARCH_KEYWORDS = (
    "price", "prices", "cost", "today", "today's", "tonight", "news", "latest",
    "current", "currently", "right now", "weather", "forecast", "stock", "stocks",
    "market", "markets", "bitcoin", "btc", "ethereum", "crypto", "gold", "silver",
    "oil", "score", "who won", "election", "release", "released", "update", "updated",
    "this week", "this year", "this month", "exchange rate", "interest rate",
    "inflation", "gdp", "earnings", "nba", "nfl", "premier league", "world cup",
)

_CONTEXT_WORDS = {
    "it", "its", "this", "that", "they", "them", "their", "he", "she", "his",
    "her", "those", "these",
}


def keyword_suggests_search(question: str) -> bool:
    q = f" {question.lower()} "
    return any(k in q for k in SEARCH_KEYWORDS)


def needs_context(question: str) -> bool:
    """True for pronoun-heavy follow-ups that need conversation history to become a
    good search query (e.g. 'and what about its price?'). A short but self-contained
    question ('bitcoin price today') does NOT need context -- length is not the signal."""
    words = set(question.lower().replace("?", " ").replace(".", " ").replace(",", " ").split())
    return bool(words & _CONTEXT_WORDS)

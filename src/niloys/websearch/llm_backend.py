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
    "changing information (news, prices, weather, scores, releases, events, rates). "
    "Reply NOSEARCH for everything else: definitions, explanations, opinions, math, "
    "how-to, casual chat, and stable general knowledge -- naming a product, asset or "
    "company does NOT by itself make a question time-sensitive.\n"
    "Answer with EXACTLY one line: SEARCH or NOSEARCH. "
    "If SEARCH, add a second line with a short web search query."
)

# Few-shot routing examples (generic, not user questions): a small model classifies far
# more reliably from examples than from instructions alone.
CLASSIFY_FEWSHOT = (
    ("What is photosynthesis?", "NOSEARCH"),
    ("Explain how compound interest works.", "NOSEARCH"),
    ("Is it safe to invest in stocks?", "NOSEARCH"),
    ("How do I change a car tire?", "NOSEARCH"),
    ("What is the capital of France?", "NOSEARCH"),
    ("What is the price of gold today?", "SEARCH\ngold price today per ounce"),
    ("Who won the match last night?", "SEARCH\nlatest match result"),
    ("What's the weather right now?", "SEARCH\nweather now"),
    ("What is the latest tech news?", "SEARCH\nlatest technology news"),
    ("What is the current exchange rate for the euro?", "SEARCH\neuro exchange rate today"),
)

ANSWER_SYSTEM = (
    "You answer the user's question using ONLY the web results provided. "
    "Reply with the answer itself, exactly as you would say it out loud: one or two short, "
    "plain spoken sentences, under {max_words} words, with the most important fact (a "
    "number, price or name) first. Do NOT add any label or prefix such as 'The most "
    "important fact is:' or 'The answer is:' -- just state the fact. No lists, no markdown, "
    "no URLs, no filler, no 'according to'. If the results do not answer the question, say "
    "you could not find it."
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
        messages = [{"role": "system", "content": CLASSIFY_SYSTEM}]
        for ex_q, ex_a in CLASSIFY_FEWSHOT:
            messages.append({"role": "user", "content": f"User's question: {ex_q}"})
            messages.append({"role": "assistant", "content": ex_a})
        messages.append({"role": "user", "content": user})
        try:
            text = self.generate(messages, max_new_tokens=24)
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


# Obvious "needs CURRENT / changing information" signals. A self-contained question with
# one of these is sent to Tavily directly (the keyword fast-path), so a conservative small
# LLM can never suppress an obvious search. Topic nouns (bitcoin, gold, ...) are NOT here
# on purpose: "what is Bitcoin?" is a static question the model answers from its own
# knowledge -- only a current-info signal (price, today, news, ...) justifies a search.
# Ambiguous / follow-up questions still go to the LLM classifier (which sees the history).
SEARCH_KEYWORDS = (
    "price", "prices", "cost", "today", "today's", "tonight", "news", "latest",
    "current", "currently", "right now", "weather", "forecast", "score", "who won",
    "election", "release", "released", "update", "updated", "this week", "this year",
    "this month", "exchange rate", "interest rate", "inflation", "gdp", "earnings",
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


# Stable-knowledge question openers (definition / explanation / opinion / how-to). A
# question that starts like this AND carries no current-info keyword is answered from the
# model's own knowledge (NOSEARCH) without ever calling the classifier -- this is the
# deterministic guard that stops "What is Bitcoin?" from triggering a web search.
_STABLE_STARTERS = (
    "what is", "what are", "what's", "whats", "who is", "who are", "who was", "who were",
    "what does", "what do", "what was", "what were", "explain", "define", "describe",
    "tell me about", "how does", "how do", "how is", "how are", "why is", "why are",
    "why does", "why do", "is it", "are they", "can you", "could you", "should i",
    "do you", "what kind", "how to",
)


def is_stable_question(question: str) -> bool:
    """True for definition / explanation / opinion / how-to questions with no current-info
    signal -- answered directly from the model's knowledge (NOSEARCH)."""
    q = " ".join(str(question or "").lower().split())
    if not q:
        return False
    if keyword_suggests_search(q):
        return False   # a current-info signal always wins ("what is the price of bitcoin?")
    return any(q.startswith(s) for s in _STABLE_STARTERS)


def split_questions(text: str) -> list:
    """Split a transcribed utterance into individual questions (the user sometimes asks
    several in one message). Splits after each '?', keeping it; falls back to the whole
    string when there is no '?'."""
    import re
    text = " ".join(str(text or "").split())
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"(?<=\?)\s+", text)]
    parts = [p for p in parts if p]
    return parts if parts else [text]

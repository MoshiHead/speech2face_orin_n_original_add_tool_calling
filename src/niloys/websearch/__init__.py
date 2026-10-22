"""Web-search augmentation for the PersonaPlex + IMTalker live server.

kyutai/stt-1b-en_fr transcribes the mic -> a small local instruct LLM (downloaded from
Hugging Face, no API; default Qwen/Qwen2.5-1.5B-Instruct) decides whether the question
needs current online information -> Tavily top-3 search -> the same LLM writes a short
answer -> the answer is fed into PersonaPlex through its native text-prompt stream so
it replies in its own voice. Any failure falls back to the classic pipeline (the model
answers from its own knowledge).
"""
from .config import WebSearchConfig
from .manager import WebSearchManager

__all__ = ["WebSearchConfig", "WebSearchManager"]

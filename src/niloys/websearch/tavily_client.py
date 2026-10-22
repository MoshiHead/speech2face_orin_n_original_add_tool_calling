"""Minimal Tavily Search API client (async aiohttp for the server, sync urllib for tools).

    POST https://api.tavily.com/search
    Authorization: Bearer <key>
    {"query": ..., "max_results": 3, "search_depth": "basic", "include_answer": false}
    -> {"results": [{"title", "url", "content", "score", ...}], "answer"?: str, ...}
"""
from __future__ import annotations

import json
import urllib.request

TAVILY_URL = "https://api.tavily.com/search"


def _payload(query: str, max_results: int, include_answer: bool) -> dict:
    return {
        "query": query,
        "max_results": int(max_results),
        "search_depth": "basic",   # "advanced" costs 2 credits and adds latency
        "topic": "general",
        "include_answer": "basic" if include_answer else False,
    }


async def tavily_search_async(
    query: str,
    api_key: str,
    max_results: int = 3,
    timeout: float = 6.0,
    include_answer: bool = False,
) -> dict:
    import aiohttp

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = _payload(query, max_results, include_answer)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=float(timeout))) as sess:
        async with sess.post(TAVILY_URL, json=body, headers=headers) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"tavily HTTP {resp.status}: {text[:200]}")
            return json.loads(text)


def tavily_search_sync(
    query: str,
    api_key: str,
    max_results: int = 3,
    timeout: float = 6.0,
    include_answer: bool = False,
) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = json.dumps(_payload(query, max_results, include_answer)).encode()
    req = urllib.request.Request(TAVILY_URL, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
        return json.loads(resp.read().decode())


def format_results(data: dict, max_results: int = 3, max_chars_each: int = 700) -> list:
    """Trim a Tavily response to the top-N (title, content, url) triples for the LLM prompt."""
    out = []
    for r in (data.get("results") or [])[:max_results]:
        title = " ".join(str(r.get("title", "")).split())[:200]
        content = " ".join(str(r.get("content", "")).split())[:max_chars_each]
        if content:
            out.append({"title": title, "content": content, "url": str(r.get("url", ""))})
    return out

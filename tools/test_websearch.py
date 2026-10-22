"""tools/test_websearch.py -- checks for the web-search augmentation (no browser, no mic).

    env/bin/python tools/test_websearch.py --stt
        transcribe tools/test_question.wav with kyutai/stt-1b-en_fr (downloads ~2.6 GB once)
    env/bin/python tools/test_websearch.py --classify "what is the bitcoin price today?"
        run the local routing LLM (Qwen2.5-1.5B-Instruct, downloads ~3.1 GB once)
    env/bin/python tools/test_websearch.py --search "bitcoin price today"
        one live Tavily call (needs TAVILY_API_KEY)
    env/bin/python tools/test_websearch.py --e2e 8991 --logfile run.log
        stream test_question.wav into the RUNNING server like a browser would, then print
        the [WEBSEARCH] lines from the log (question -> classify -> tavily -> answer ->
        injection). PASS = the plumbing worked end to end (NOSEARCH is also a valid
        classification -- it means the recorded question simply needs no web search).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "niloys"))
sys.path.insert(0, os.path.join(ROOT, "src", "moshi_pkg"))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf"))

SR, FRAME = 24000, 1920


def _load_test_wav(seconds: float = 30.0):
    import numpy as np
    import torchaudio

    wav, sr = torchaudio.load(os.path.join(HERE, "test_question.wav"))
    wav = wav.mean(0) if wav.shape[0] > 1 else wav[0]
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    p = wav.numpy()
    need = int(seconds * SR)
    if len(p) < need:
        p = np.concatenate([p, 1e-4 * np.random.randn(need - len(p)).astype(np.float32)])
    return p[:need].astype("float32")


def cmd_stt(args) -> int:
    import torch

    from websearch.stt_engine import KyutaiSTT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] loading STT {args.stt_repo} on {device} ...", flush=True)
    t0 = time.perf_counter()
    stt = KyutaiSTT(args.stt_repo, device, int4=args.stt_int4)
    print(f"[test] loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    pcm = _load_test_wav(30.0)
    t0 = time.perf_counter()
    text = stt.transcribe_offline(pcm)
    dt = time.perf_counter() - t0
    print(f"[test] STT transcript ({dt:.1f}s for {len(pcm) / SR:.1f}s of audio):")
    print(f"       {text!r}")
    ok = len(text.split()) >= 3
    print("PASS" if ok else "FAIL: transcript too short -- check the model load messages above")
    return 0 if ok else 1


def cmd_classify(args) -> int:
    import torch

    from websearch.llm_backend import LocalLLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] loading LLM {args.llm_model} on {device} ...", flush=True)
    llm = LocalLLM(args.llm_model, device)
    question = " ".join(args.question)
    t0 = time.perf_counter()
    needs, query = llm.classify(question)
    print(f"[test] classify ({time.perf_counter() - t0:.2f}s): "
          f"{'SEARCH' if needs else 'NOSEARCH'}  query={query!r}")
    if needs:
        results = [
            {"title": "Example result", "content": "Bitcoin is trading at about $100,000 today."},
            {"title": "Second result", "content": "Markets were mixed this morning."},
        ]
        answer = llm.synthesize(question, results)
        print(f"[test] synthesized answer: {answer!r}")
    print("PASS (classification ran; SEARCH/NOSEARCH are both valid outcomes)")
    return 0


def cmd_search(args) -> int:
    from websearch.tavily_client import format_results, tavily_search_sync

    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        print("FAIL: TAVILY_API_KEY is not set")
        return 1
    query = " ".join(args.query)
    t0 = time.perf_counter()
    data = tavily_search_sync(query, key, max_results=3)
    results = format_results(data, 3)
    print(f"[test] tavily ({time.perf_counter() - t0:.2f}s): {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['title']}\n      {r['content'][:160]}")
    ok = len(results) >= 1
    print("PASS" if ok else "FAIL: no results")
    return 0 if ok else 1


def cmd_e2e(args) -> int:
    import asyncio

    import numpy as np
    import websockets

    pcm = _load_test_wav(30.0)
    i16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)

    async def run() -> None:
        ws = await websockets.connect(
            f"ws://127.0.0.1:{args.port}/ws/conversation", max_size=None, ping_interval=None
        )
        ready = json.loads(await ws.recv())
        vws = await websockets.connect(
            f"ws://127.0.0.1:{args.port}{ready.get('video_ws_path')}",
            max_size=None, ping_interval=None,
        )

        async def drain(sock):
            try:
                async for _ in sock:
                    pass
            except Exception:
                pass

        tasks = [asyncio.create_task(drain(ws)), asyncio.create_task(drain(vws))]
        await ws.send(json.dumps({"type": "start", "sample_rate": SR}))
        await asyncio.sleep(1.0)
        t0 = time.perf_counter()
        n = len(i16) // FRAME
        for i in range(n):
            await ws.send(b"\x03" + i16[i * FRAME:(i + 1) * FRAME].tobytes())
            dt = t0 + (i + 1) * FRAME / SR - time.perf_counter()
            if dt > 0:
                await asyncio.sleep(dt)
        print("[test] audio streamed; waiting 12s for the search pipeline ...", flush=True)
        await asyncio.sleep(12)
        for t in tasks:
            t.cancel()
        for s in (ws, vws):
            try:
                await s.close()
            except Exception:
                pass

    asyncio.run(run())
    time.sleep(1)
    lines = []
    if os.path.exists(args.logfile):
        with open(args.logfile, errors="replace") as fh:
            lines = [l.rstrip() for l in fh if "[WEBSEARCH]" in l]
    print(f"[test] {len(lines)} [WEBSEARCH] log lines in {args.logfile}:")
    for l in lines[-25:]:
        print("   ", l[:220])
    heard = any("question:" in l for l in lines)
    classified = any("classify" in l for l in lines)
    injected = any("injection:" in l for l in lines) or any("DRY RUN" in l for l in lines)
    nosearch = any("NOSEARCH" in l for l in lines)
    ok = heard and classified and (injected or nosearch)
    print(f"[test] heard_question={heard} classified={classified} "
          f"injected={injected} nosearch={nosearch}")
    print("PASS" if ok else "FAIL: expected the STT question + a classification "
          "(+ an injection unless the question needed no search)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stt", action="store_true", help="transcribe test_question.wav with the STT model")
    ap.add_argument("--classify", nargs=argparse.REMAINDER, help="run the routing LLM on a question")
    ap.add_argument("--search", nargs=argparse.REMAINDER, help="one live Tavily call")
    ap.add_argument("--e2e", type=int, default=0, metavar="PORT", help="full turn against a running server")
    ap.add_argument("--logfile", default=os.path.join(ROOT, "run.log"))
    ap.add_argument("--stt_repo", default="kyutai/stt-1b-en_fr")
    ap.add_argument("--stt_int4", action="store_true")
    ap.add_argument("--llm_model", default=os.environ.get("WEBSEARCH_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    args = ap.parse_args()
    if args.stt:
        return cmd_stt(args)
    if args.classify:
        return cmd_classify(args)
    if args.search:
        return cmd_search(args)
    if args.e2e:
        return cmd_e2e(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())


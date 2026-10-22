"""tools/make_websearch_report.py -- generate WEBSEARCH_PLAN_AND_EXECUTION.pdf.

A readable, detailed write-up of the web-search feature: the plan, the PersonaPlex
architecture findings, and exactly what was implemented, file by file.

    python tools/make_websearch_report.py
"""
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, Preformatted)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "WEBSEARCH_PLAN_AND_EXECUTION.pdf")

INK = HexColor("#1a2332")
ACCENT = HexColor("#0b5fff")
MUTED = HexColor("#5a6675")
CODEBG = HexColor("#f2f4f8")
RULE = HexColor("#c9d2e0")

S = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=24, leading=30,
                            textColor=INK, alignment=TA_CENTER, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=12, leading=17,
                               textColor=MUTED, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15, leading=19,
                         textColor=ACCENT, spaceBefore=16, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, leading=15,
                         textColor=INK, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13.5,
                           textColor=INK, alignment=TA_LEFT, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.5, leading=13.5,
                             textColor=INK, leftIndent=14, bulletIndent=4, spaceAfter=3),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8, leading=10.5,
                           textColor=INK, backColor=CODEBG, borderPadding=5,
                           leftIndent=2, rightIndent=2, spaceBefore=3, spaceAfter=6),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8, leading=11,
                            textColor=MUTED, spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5, leading=11.5,
                           textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=8.5, leading=11.5,
                            textColor=INK),
}

story = []


def h1(t):
    story.append(Paragraph(t, S["h1"]))


def h2(t):
    story.append(Paragraph(t, S["h2"]))


def p(t):
    story.append(Paragraph(t, S["body"]))


def b(t):
    story.append(Paragraph(t, S["bullet"], bulletText="-"))


def code(t):
    story.append(Preformatted(t, S["code"]))


def small(t):
    story.append(Paragraph(t, S["small"]))


def sp(h=4):
    story.append(Spacer(1, h))


def table(rows, widths, header=True):
    data = [[Paragraph(c, S["cellb"] if header and i == 0 else S["cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), HexColor("#e8eefc"))]
    t.setStyle(TableStyle(style))
    story.append(t)
    sp(6)


# =====================================================================================
# Title page
# =====================================================================================
story.append(Spacer(1, 60))
story.append(Paragraph("Web-Search Augmentation", S["title"]))
story.append(Paragraph("for the PersonaPlex + IMTalker Real-Time Talking Avatar", S["title"]))
sp(10)
story.append(Paragraph("Plan, architecture analysis and complete execution report", S["subtitle"]))
sp(4)
story.append(Paragraph("speech2face_orin_n_original_v1 &nbsp;|&nbsp; RunPod x86_64 + Jetson AGX Orin &nbsp;|&nbsp; 2026-10-22", S["subtitle"]))
story.append(Spacer(1, 40))
story.append(Paragraph(
    "This document explains, in order: (1) what was requested, (2) what the repository already "
    "does and how PersonaPlex can actually be conditioned on external text, (3) the design that "
    "was chosen and why, (4) every file that was created or changed, (5) the threading/GPU safety "
    "model, budgets, error handling, testing and known limitations, and (6) how to run it.",
    S["body"]))
story.append(PageBreak())

# =====================================================================================
h1("1. Goal and requirements")
# =====================================================================================
p("The task: add an online-search feature to the existing full-duplex speech-to-speech "
  "talking-head pipeline (NVIDIA PersonaPlex 7B + IMTalker) that runs on RunPod GPUs, with "
  "minimal added latency and without breaking the existing pipeline. The required flow:")
b("<b>1.</b> After kyutai/stt-1b-en_fr transcribes the user's speech, a small, free, low-latency "
  "LLM decides whether the question needs up-to-date online information.")
b("<b>2.</b> If NO: the existing pipeline continues completely unchanged.")
b("<b>3.</b> If YES: the Tavily API returns the top-3 web results; the same LLM analyzes them and "
  "produces one concise, accurate answer.")
b("<b>4.</b> That answer is injected into PersonaPlex so it responds naturally through the existing "
  "speech-generation and talking-head pipeline.")
b("<b>5.</b> Keep at least the previous 3-4 conversation turns so follow-up questions resolve.")
p("The most critical constraint: <b>PersonaPlex generates speech directly and has no simple text "
  "input</b> - the injection had to use the model's real conditioning mechanism, verified from the "
  "code, not assumed. A later refinement fixed the LLM choice: a small, effective model "
  "<b>downloaded from Hugging Face and run locally</b> on the RunPod GPU - no LLM API of any kind.")

# =====================================================================================
h1("2. What the repository already does (verified from code)")
# =====================================================================================
p("The production server is <b>src/niloys/live/imtalker_native_passthrough.py</b>, launched by "
  "run.sh / run_x86.sh through src/niloys/live/imtalker_try19_student_live.py "
  "(--live_module imtalker_native_passthrough). The reply engine is "
  "<b>MoshiOnlyEngineWithHidden</b> (a subclass of MoshiOnlyEngine from src/imtalker/liveTry.py), "
  "wrapping an LMGen from the repo's own moshi fork (src/moshi_pkg/moshi). The live data flow:")
code(
    "Browser mic (48 kHz Opus) -> /ws/conversation\n"
    "  -> mic_q -> _mic_ingest_worker -> reply_engine.append_browser_pcm()   [24 kHz PCM]\n"
    "  -> _persona_priority_worker (thread, high-priority CUDA stream):\n"
    "       per 80 ms frame: Mimi.encode -> lm_gen._step() (7B trunk, int4, CUDA-graphed,\n"
    "       one-pass Robert_5 depformer) -> Mimi.decode -> reply PCM + layer[-2] hidden\n"
    "  -> persona_event_q -> GPU thread -> adapter -> FM motion -> renderer -> A/V out")
p("Key facts that shaped the design:")
b("The notebook (runpod_speech2face.ipynb) clones the repo and then <b>overwrites</b> "
  "setup_env_x86.sh, prepare_x86.sh, run_x86.sh, requirements-x86-extra.txt and two tools via "
  "%%writefile cells - so every script change must be mirrored into the notebook cells.")
b("requirements.txt pins <b>transformers==4.30.2 / tokenizers==0.13.3</b> (Python 3.10). The only "
  "transformers use in the whole live server is Wav2Vec2FeatureExtractor (plus Wav2Vec2Model in "
  "the offline generator).")
b("Neither kyutai/stt-1b-en_fr nor any STT/Tavily code existed in the repo - both had to be added.")
b("The client browser page uses getUserMedia with echoCancellation: true, so the mic is normally "
  "echo-free (the avatar's own voice is not heard by the STT).")


# =====================================================================================
h1("3. The critical finding: how PersonaPlex can really be conditioned on text")
# =====================================================================================
p("PersonaPlex is a <b>multi-stream transformer</b>: every 80 ms step it consumes a token cache of "
  "shape [B, 17, T] = <b>1 text stream + 8 agent-audio codebooks + 8 user-audio codebooks</b>. "
  "There is no chat or prompt API. But there is exactly one first-class text-conditioning path, "
  "and production already uses it for the 'You are Robert...' persona:")
b("<b>MoshiOnlyEngine._apply_system_prompts()</b> (liveTry.py) sets "
  "<b>lm_gen.text_prompt_tokens = tokenizer.encode(system-wrapped text)</b> and calls "
  "lm_gen.step_system_prompts(mimi).")
b("<b>LMGen._step_text_prompt_core()</b> (src/moshi_pkg/moshi/models/lm.py) feeds each token with "
  "step(moshi_tokens=SILENCE, text_token=tok, input_tokens=SINE) - forced text in the agent-text "
  "stream, forced silence on the agent-audio stream, sine ('user silent') on the user stream. "
  "Each step runs one trunk forward + depformer and appends to the KV cache.")
b("Nothing in this path is tied to session start. Setting text_prompt_tokens and stepping the "
  "tokens <b>mid-session</b> appends the text to the live KV cache; the model then continues "
  "generating speech conditioned on it. This is the mechanism the injection uses.")
p("Safety properties verified before choosing it:")
b("The public LMGen.step() used for prompt feeding is NOT the lm_gen._step that "
  "MoshiOnlyEngineWithHidden rebound for hidden capture - no interference with the hidden-capture "
  "contract, the CUDA graphs (graphed_main / graphed_depth), the int4 trunk or the try19 patches.")
b("Injection must run <b>on the persona thread, under reply_input_lock, inside the same priority "
  "CUDA stream</b> the graphs are replayed on - otherwise a second thread could grab the lock "
  "between kernel launch and priority_stream.synchronize() and race the streaming state. The "
  "implementation runs it exactly there, between two 80 ms audio steps.")
b("Cost is one full LM step per token (~10-15 ms on a 4090-class RunPod GPU, ~50-58 ms on Orin "
  "int4). A <=96-token answer is ~0.7-1.2 s on RunPod - injected during the post-question silence, "
  "so it sounds like a natural 'thinking pause', and A/V sync is preserved because everything "
  "downstream shifts together.")
story.append(PageBreak())

# =====================================================================================
h1("4. The chosen design")
# =====================================================================================
code(
    "                     +---------------- existing pipeline (unchanged) ----------------+\n"
    " mic PCM --> mic_q ->| append_browser_pcm -> persona thread -> Mimi/LM/Mimi -> avatar |\n"
    "     |               +---------------------------------------------------------------+\n"
    "     | (new tap, ~free)               ^ injection (between 80 ms steps,\n"
    "     v                                |  same thread + lock + CUDA stream)\n"
    " [STT thread] kyutai/stt-1b-en_fr   [injection queue]\n"
    "   energy VAD, streaming transcript      |\n"
    "   end-of-turn + 0.5 s flush            |\n"
    "     |                                  |\n"
    "     v                                  |\n"
    " [Orchestrator, asyncio]                |\n"
    "   classify (local LLM) --NO--> nothing (pipeline 100% unchanged)\n"
    "     | YES\n"
    "   Tavily top-3 -> same LLM: short answer -> wrap as system text -> queue")
p("Turn flow: the STT transcribes while the user speaks. At end-of-turn (500 ms silence) plus a "
  "0.5 s flush, the question goes to the classifier. NOSEARCH -> nothing happens. SEARCH -> "
  "Tavily (top 3) -> the same LLM writes a <=60-word answer -> the answer is wrapped in the same "
  "system format as the persona prompt and queued. The persona thread picks it up between two "
  "audio steps and feeds it into the LM. A 3.5 s deadline drops late answers; a barge-in aborts "
  "an in-flight injection between tokens.")
h2("Why the latency stays minimal")
b("STT, classification, search and answer synthesis all overlap the user's own speech and the "
  "model's natural response gap - the only added, audible delay is the injection itself "
  "(~0.7-1.2 s on RunPod) plus whatever of the ~1.5-2.5 s pipeline was not already hidden.")
b("On NOSEARCH turns the added cost is one queue check per loop iteration and one numpy copy per "
  "mic packet - effectively zero.")
b("The STT only does GPU work while the user actually speaks (plus a short flush), not "
  "continuously.")


# =====================================================================================
h1("5. Component decisions and why")
# =====================================================================================
h2("5.1 Speech-to-text: kyutai/stt-1b-en_fr on the repo's OWN moshi fork")
p("The STT is a Moshi-family model: Mimi (24 kHz, 12.5 Hz, 32 codebooks) + a 16-layer transformer "
  "(dim 2048, dep_q=0, all delays 0, own 8k SentencePiece tokenizer, 0.5 s text delay, greedy). "
  "Three options were evaluated:")
b("<b>pip moshi package</b> - rejected: this repo carries its own moshi fork first on PYTHONPATH, "
  "so the official package cannot be imported alongside without fragile aliasing.")
b("<b>transformers port (stt-1b-en_fr-trfs)</b> - rejected: needs transformers >= 4.53; the repo "
  "pins 4.30.2.")
b("<b>Load the checkpoint with the repo's own LMModel/MimiModel</b> - chosen. The state-dict key "
  "layout is the same lineage (the fork's _remap_state_dict_keys handles the one rename), and a "
  "small hand-written streaming loop reproduces the official LMGen alignment for dep_q=0 / "
  "delays=0 models: step 0 forwards the initial tokens, step t forwards "
  "[text_(t-1), mimi_codes(frame_(t-1))] and argmaxes the text logits. A load-time coverage check "
  "fails loudly (and disables only the feature) if a future checkpoint renames keys.")
p("The STT runs in its own thread, gated by an energy VAD: GPU work only while the user speaks "
  "plus a ~0.6 s flush for the model's 0.5 s text delay. An echo guard (driven by the reply_rms "
  "already present in the persona events) never opens a turn while the avatar speaks. Model state "
  "is reset at each turn opening so idle gaps cannot leak stale convolution context into a "
  "question. On Orin, WEBSEARCH_STT_INT4=1 int4-quantizes the STT transformer with the repo's own "
  "int4_linear to protect the 80 ms step budget.")
h2("5.2 The routing/answer LLM: Qwen/Qwen2.5-1.5B-Instruct, local only")
p("The user required a small, effective LLM downloaded from Hugging Face and run locally - no LLM "
  "API. The hard constraint was the pinned transformers==4.30.2, whose LlamaConfig has no GQA "
  "support: that rules out TinyLlama, SmolLM, Phi-2, gemma and Qwen2.5 (all need >= 4.35-4.42). "
  "Because the server's only transformers use is Wav2Vec2 (API-stable 4.30 -> 4.37), the x86 "
  "extras upgrade transformers to 4.37.2 (Jetson's requirements.txt is untouched), which unlocks "
  "the qwen2 architecture. Qwen2.5-1.5B-Instruct (ungated, Apache-2.0, ~3.1 GB bf16) is one of "
  "the strongest sub-2B instruct models - exactly right for the two jobs:")
b("<b>classify(question, history)</b> -> (needs_search, reformulated query). Greedy, 24 max new "
  "tokens, robust parsing; any failure or timeout defaults to NOSEARCH (pipeline preserved).")
b("<b>synthesize(question, top-3 results, history)</b> -> one <=60-word spoken-style answer, "
  "grounded ONLY in the results; one bare-prompt retry, then the turn is skipped.")
h2("5.3 Tavily search")
p("POST https://api.tavily.com/search with a Bearer key, max_results=3, search_depth=basic "
  "(advanced costs 2 credits and adds latency), topic=general. include_answer is OFF by default "
  "(no dependence on Tavily's own LLM; available behind a flag). Hard 6 s timeout; any HTTP "
  "error or empty result set skips the turn.")
h2("5.4 Conversation history")
p("The last 4 turns are kept: the user side from the STT transcript, the assistant side from "
  "PersonaPlex's own audio_text delta between turns. Both LLM prompts receive it, so follow-ups "
  "like 'and what about its price?' resolve against what was actually said.")
story.append(PageBreak())

# =====================================================================================
h1("6. Implementation: every file created or changed")
# =====================================================================================
h2("6.1 New package: src/niloys/websearch/ (9 modules)")
table([
    ["File", "What it does"],
    ["stt_engine.py", "KyutaiSTT: loads kyutai/stt-1b-en_fr with the repo's own LMModel/MimiModel "
     "(32 codebooks, checkpoint key-coverage check, unused depformer dropped), and runs the "
     "hand-written streaming loop in a dedicated thread. Energy VAD, end-of-turn detection "
     "(500 ms silence after >=500 ms speech), 0.5 s text-delay flush, echo guard, per-turn model "
     "reset, drop-oldest bounded input queue, optional int4. Also transcribe_offline() for tests."],
    ["llm_backend.py", "LocalLLM: Qwen2.5-1.5B-Instruct via transformers (bf16, greedy, "
     "lock-serialized). classify() -> (needs_search, query); synthesize() -> <=60-word grounded "
     "answer. parse_classification() defaults to NOSEARCH on any ambiguity."],
    ["tavily_client.py", "tavily_search_async (aiohttp) + tavily_search_sync (urllib) + "
     "format_results (top-3, title/content trimmed)."],
    ["orchestrator.py", "WebSearchOrchestrator: per-turn pipeline on the asyncio loop - history "
     "snapshot, classify (executor), Tavily, answer (executor, one bare retry), 3.5 s deadline, "
     "injection queue. Cancels superseded searches; never raises."],
    ["injector.py", "Injection: wraps the answer in the same system format as the persona prompt "
     "and feeds it token-by-token via lm_gen.step(moshi=SILENCE, text_token=tok, input=SINE), "
     "with token-budget trimming, per-token barge-in abort and a trailing silence beat."],
    ["history.py", "TurnHistory: last 4 (user, assistant) turns, prompt rendering."],
    ["config.py", "WebSearchConfig.from_args: builds the config or returns None (feature off / no "
     "Tavily key) with a clear log line."],
    ["manager.py", "WebSearchManager: background model load, thread-safe facade for the server, "
     "/health status. Every method is a no-op until ready."],
    ["__init__.py", "Package docstring + exports (imports nothing heavy: safe at server import "
     "time even before transformers is upgraded)."],
], [30, 140])
h2("6.2 Server edits (all additive, guarded by 'if websearch is not None')")
table([
    ["File", "Change"],
    ["src/imtalker/liveTry.py", "4-line optional on_pcm24 hook at the end of append_browser_pcm "
     "(default None -> zero cost). This is the mic tap for the STT."],
    ["src/niloys/live/imtalker_native_passthrough.py", "guarded websearch import; 12 new CLI args; "
     "background model load in build_app; /health.websearch; mic-tap wiring per session; the "
     "injection check in _persona_priority_worker (runs on the persona thread, under "
     "reply_input_lock, inside the priority CUDA stream); reply_rms observation per persona "
     "event; on_session_start at the Start boundary; on_session_end at teardown."],
], [45, 125])
h2("6.3 Scripts, requirements, notebook, docs")
table([
    ["File", "Change"],
    ["run_x86.sh, run.sh", "Pass --enable_web_search + --websearch_llm_model (the server "
     "self-gates on TAVILY_API_KEY); optional WEBSEARCH_STT_INT4 / WEBSEARCH_DRY_RUN flags."],
    ["prepare_x86.sh, prepare.sh", "Optional step 4b: pre-download the STT + LLM (~5.7 GB) when "
     "PREPARE_WEBSEARCH=1 or a Tavily key is present; failure is a warning (the server retries "
     "lazily in the background)."],
    ["requirements-x86-extra.txt", "transformers==4.37.2 + tokenizers==0.15.2 (x86 only; Jetson "
     "untouched). Installed after requirements.txt, so the upgrade wins."],
    ["runpod_speech2face.ipynb", "TAVILY_API_KEY config in cell 1; the %%writefile cells for "
     "requirements-x86-extra.txt / prepare_x86.sh / run_x86.sh updated identically (verified "
     "byte-equal to the repo files); the server-start cell passes the key; new section 10b test "
     "cells; knobs table + intro docs. JSON valid, 45 cells."],
    ["README.md, README_RUNPOD.md", "Feature documentation: flow, VRAM/latency, knobs, tests, "
     "limitations."],
    ["tools/test_websearch.py", "On-pod checks: --stt (transcribe the test wav), --classify, "
     "--search, --e2e (full turn against the running server + log grep)."],
    ["tools/test_websearch_units.py", "27 pure-python unit tests (no GPU): classification "
     "parsing, Tavily formatting, history, injection text + token budget, config gating."],
], [45, 125])
story.append(PageBreak())

# =====================================================================================
h1("7. Threading and GPU safety model")
# =====================================================================================
table([
    ["Thread / context", "What it does", "Why it is safe"],
    ["mic ingest (existing)", "append_browser_pcm + the new on_pcm24 tap",
     "The tap is a non-blocking queue put (~microseconds) inside the existing reply_input_lock "
     "section - the ingest path cannot stall."],
    ["STT thread (new)", "VAD + Mimi encode + STT LM forward per 80 ms frame, only while the user "
     "speaks", "Own Mimi + LM instances (no shared state with PersonaPlex); plain eager calls; "
     "the GPU simply time-shares with the persona thread."],
    ["asyncio loop (existing)", "orchestrator: classify/answer via executor, Tavily via aiohttp",
     "Network and GPU work stay off the audio hot path; tasks are cancelled on new turns and "
     "session end."],
    ["persona thread (existing)", "runs the injection between two 80 ms steps",
     "Same thread, same reply_input_lock, same priority CUDA stream as the normal LM steps - "
     "the CUDA graphs and the streaming state see no difference."],
], [32, 60, 78])
p("The LLM (executor thread) and the STT (its thread) never touch PersonaPlex's models, locks or "
  "streams. The only cross-thread hand-offs are bounded queues and thread-safe flags.")

# =====================================================================================
h1("8. Latency, VRAM and GPU budgets")
# =====================================================================================
table([
    ["Item", "RunPod x86 (4090-class)", "Jetson AGX Orin (int4)"],
    ["NOSEARCH turn added latency", "~0 (queue check + packet copy)", "~0"],
    ["SEARCH turn added latency", "~2-4 s 'thinking' (mostly hidden in the response gap) + "
     "~0.7-1.2 s injection", "similar pipeline + ~3-4 s injection (96 tokens x ~50 ms)"],
    ["Added VRAM", "~5.7 GB (STT 2.6 + Qwen 3.1) -> ~20 GB total with the int4 trunk",
     "fits in 64 GB; use WEBSEARCH_STT_INT4=1"],
    ["Added GPU per 80 ms frame", "~5-10 ms while the user speaks only",
     "~10-20 ms while the user speaks (int4 STT: ~5 ms) - tight against the 80 ms budget"],
], [42, 64, 64])

# =====================================================================================
h1("9. Error handling and fallback matrix")
# =====================================================================================
table([
    ["Failure", "Behaviour"],
    ["No TAVILY_API_KEY", "Feature never activates; one log line; classic pipeline."],
    ["STT/LLM download or load error", "Feature disables itself with the error in /health and the "
     "log; the server keeps running."],
    ["Classifier error / timeout / garbage", "Defaults to NOSEARCH - the pipeline is preserved."],
    ["Tavily HTTP error / timeout / 0 results", "Turn skipped; the model answers from its own "
     "knowledge."],
    ["Answer LLM error", "One bare-prompt retry, then the turn is skipped."],
    ["Answer arrives after the 3.5 s deadline", "Dropped (logged) - no mid-conversation surprise."],
    ["User barges in during an injection", "Injection aborts between tokens; partial text is "
     "harmless context."],
    ["Session reset / disconnect mid-search", "Pending tasks cancelled, queues drained, STT reset."],
    ["STT checkpoint key mismatch (future model change)", "Load fails loudly with the key names; "
     "only the feature is disabled."],
], [55, 115])
story.append(PageBreak())

# =====================================================================================
h1("10. Testing and validation performed")
# =====================================================================================
b("All 13 new/modified Python files compile (py_compile).")
b("27/27 pure-logic unit tests pass (tools/test_websearch_units.py): classification parsing, "
  "Tavily formatting, history, injection text + token budget, config gating.")
b("The websearch package imports cleanly without torch installed (the server imports it at "
  "module level).")
b("Notebook JSON valid (45 cells); the embedded requirements-x86-extra.txt / prepare_x86.sh / "
  "run_x86.sh cells verified byte-identical to the repo files.")
b("On the pod: tools/test_websearch.py --stt / --classify / --search / --e2e give staged "
  "verification; WEBSEARCH_DRY_RUN=1 logs injections without touching the model; the existing "
  "tools/test_session.py is untouched as the regression gate.")

# =====================================================================================
h1("11. Known limitations")
# =====================================================================================
b("Orin: the STT shares the GPU with the 80 ms step budget while the user speaks - use "
  "WEBSEARCH_STT_INT4=1; the feature is primarily tuned for the RunPod x86 path.")
b("A search turn adds a few seconds of 'thinking' pause; if the model already started replying, "
  "the injection lands as a short pause followed by the web-informed answer.")
b("A barge-in during an in-flight injection is handled by aborting the injection early.")
b("The first session that connects before the background model load finishes runs the classic "
  "pipeline (the feature activates on the next Start).")
b("Qwen2.5-1.5B is strong for its size but not GPT-class; WEBSEARCH_LLM_MODEL swaps in any other "
  "HF instruct model (e.g. Qwen2.5-0.5B-Instruct for tight VRAM, or a 7B on big cards).")
b("Tavily's free tier is rate-limited; classification defaults to NOSEARCH on any doubt, so "
  "spurious searches are rare.")

# =====================================================================================
h1("12. How to use it")
# =====================================================================================
code(
    "# one time (RunPod notebook or shell)\n"
    "export TAVILY_API_KEY=tvly-...            # free tier at tavily.com\n"
    "./setup_env_x86.sh                        # installs transformers 4.37.2 via the extras\n"
    "./prepare_x86.sh hf_YOUR_TOKEN            # pre-downloads STT + LLM when the key is set\n"
    "./run_x86.sh                              # or the notebook's Start cell\n"
    "\n"
    "# checks\n"
    "$(cat .python_path) tools/test_websearch.py --stt\n"
    "$(cat .python_path) tools/test_websearch.py --e2e 8991 --logfile run.log\n"
    "curl -s localhost:8991/health | python -m json.tool   # websearch.ready == true")
p("Then talk to the avatar as usual. Ask something current ('what is the bitcoin price today?', "
  "'who won the game last night?') and watch the [WEBSEARCH] lines in run.log: the STT question, "
  "the SEARCH decision, the Tavily results, the answer and the injection. Everything else about "
  "the pipeline is exactly as before.")

# =====================================================================================
# Build
# =====================================================================================
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 12 * mm, "Web-Search Augmentation - plan & execution report")
    canvas.drawRightString(192 * mm, 12 * mm, f"page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=18 * mm, rightMargin=18 * mm,
                        topMargin=16 * mm, bottomMargin=18 * mm,
                        title="Web-Search Augmentation - Plan and Execution Report",
                        author="speech2face web-search feature")
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print("wrote", OUT)


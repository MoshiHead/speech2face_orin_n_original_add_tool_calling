"""tools/make_websearch_notebook.py -- generate runpod_speech2face_websearch.ipynb.

A RunPod notebook (same style as runpod_speech2face.ipynb) that clones the tool-calling
fork (speech2face_orin_n_original_add_tool_calling) and runs the web-search pipeline:
STT -> local LLM -> Tavily -> PersonaPlex injection, with the timestamped stage logs.

    python tools/make_websearch_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "runpod_speech2face_websearch.ipynb")

cells = []


def md(cid, text):
    cells.append({
        "cell_type": "markdown", "id": cid, "metadata": {},
        "source": [l + "\n" for l in text.split("\n")],
    })


def code(cid, text):
    cells.append({
        "cell_type": "code", "execution_count": None, "id": cid, "metadata": {},
        "outputs": [], "source": [l + "\n" for l in text.split("\n")],
    })


# =====================================================================================
md("cell-00", """# speech2face on RunPod — PersonaPlex 7B + IMTalker + **web search**
## (STT → local LLM → Tavily → PersonaPlex injection)

Full-duplex speech-to-speech (NVIDIA PersonaPlex 7B) driving a lip-synced talking head in the
browser — now with an **online-search turn**: the avatar transcribes your question with
`kyutai/stt-1b-en_fr`, a small **local** LLM (`Qwen/Qwen2.5-1.5B-Instruct`, downloaded from
Hugging Face — **no LLM API**) decides whether it needs current online information, searches
**Tavily** (top 3), and injects a short answer into PersonaPlex, which speaks it in its own voice.

This notebook clones the **tool-calling fork**
[`MoshiHead/speech2face_orin_n_original_add_tool_calling`](https://github.com/MoshiHead/speech2face_orin_n_original_add_tool_calling)
and runs it on **any x86_64 NVIDIA GPU** (5090, 4090, A100, H100, L40S, A6000 …).

### Before you start

| | |
|---|---|
| **GPU** | compute capability ≥ 8.0 (Ampere or newer). ~20 GB VRAM with the int4 trunk + STT + LLM; ~26 GB+ if this card cannot run ATen's int4 kernels (the notebook detects that and tells you) |
| **Disk** | **≥ 70 GB** on `/workspace` (weights ~19 GB, web-search models ~5.7 GB, Python env ~8 GB, HF cache) |
| **Pod template** | a RunPod *PyTorch* or *CUDA 12* image is fine — the notebook builds its own Python 3.10 env |
| **Exposed port** | add **8991** to the pod's *HTTP ports* for the RunPod proxy URL, or use the Cloudflare tunnel cell (no pod configuration) |
| **HF token** | a Hugging Face token whose account has **accepted the licence of `nvidia/personaplex-7b-v1`** (gated — accept it on the model page first) |
| **Tavily API key** | free tier at tavily.com — **required for the web-search feature** (without it the classic pipeline runs completely unchanged) |

### What the cells do

1. configuration (HF token + Tavily key)
2. check the GPU, driver and disk
3. install the system libraries
4. clone the tool-calling repo into `/workspace/speech2face_toolcalling`
5. `setup_env_x86.sh` — Python 3.10 env in `./env` (installs transformers 4.37.2 for the LLM)
6. `prepare_x86.sh` — ~19 GB of weights + the web-search models, GPU probe, TensorRT engine
7. start the server, wait for it to load, print an **HTTPS** URL (the microphone needs HTTPS)
8. headless end-to-end check + the web-search pipeline check

Total first run: roughly 45–75 minutes, almost all of it downloading. Everything lands on
`/workspace`, so a restarted pod with the same volume skips steps 5 and 6.""")

# =====================================================================================
md("cell-01", """## 1 · Configuration

The things you normally change: the Hugging Face token and the **Tavily API key** (for web
search). Nothing is hard-coded — both come from env vars, files, or a prompt.""")

code("cell-02", '''import os, subprocess, sys, time, json, shutil, urllib.request, getpass

# ----------------------------------------------------------------- settings
REPO_URL  = "https://github.com/MoshiHead/speech2face_orin_n_original_add_tool_calling.git"
REPO_DIR  = "/workspace/speech2face_toolcalling"   # the clone lands here
BRANCH    = ""                            # "" = the repo's default branch
PORT      = 8991                          # 8998 is refused by the server
VOICE     = "Robert_5.pt"                 # the depformer and decoder are trained for Robert_5
TEXT_PROMPT = ("You are Robert, an assistant from RB Labs. You have knowledge about crypto, "
               "finance, investment, technology, and general topics. Answer every question "
               "completely, descriptively, and with useful detail.")

# Hugging Face token, in order of preference:
#   1. the HF_TOKEN environment variable (set it in the pod's env vars - best on a shared pod)
#   2. /workspace/hf_token.txt
#   3. a prompt, below
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN and os.path.exists("/workspace/hf_token.txt"):
    HF_TOKEN = open("/workspace/hf_token.txt").read().strip()
if not HF_TOKEN:
    HF_TOKEN = getpass.getpass("Hugging Face token (hf_...): ").strip()
assert HF_TOKEN.startswith("hf_"), "that does not look like a Hugging Face token"
os.environ["HF_TOKEN"] = HF_TOKEN

# ----------------------------------------------------------------- web search
# Tavily API key (free tier at tavily.com). With a key, the avatar transcribes each question
# with kyutai/stt-1b-en_fr, a small LOCAL LLM (downloaded from Hugging Face - no LLM API)
# decides whether it needs current online information, searches Tavily (top 3), and injects
# a short answer into PersonaPlex. Without a key the classic pipeline runs unchanged.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
if not TAVILY_API_KEY and os.path.exists("/workspace/tavily_api_key.txt"):
    TAVILY_API_KEY = open("/workspace/tavily_api_key.txt").read().strip()
WEBSEARCH_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    os.environ["PREPARE_WEBSEARCH"] = "1"   # prepare_x86.sh pre-downloads the STT + LLM
    print("web search: ENABLED (Tavily key set, llm=%s)" % WEBSEARCH_LLM_MODEL)
else:
    print("web search: disabled (no TAVILY_API_KEY - the classic pipeline runs unchanged)")

# ----------------------------------------------------------------- helper
def sh(cmd, cwd=None, check=True, env=None, quiet=False):
    """Run a shell command, streaming its output into the notebook as it happens."""
    cwd = cwd or (REPO_DIR if os.path.isdir(REPO_DIR) else os.getcwd())
    if not quiet:
        print("$ " + cmd, flush=True)
    p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                         errors="replace", env={**os.environ, **(env or {})})
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if check and rc != 0:
        raise RuntimeError("command failed (exit %d): %s" % (rc, cmd))
    return rc

print("repo   :", REPO_URL)
print("into   :", REPO_DIR)
print("port   :", PORT)
print("token  : %s...%s  (%d chars)" % (HF_TOKEN[:6], HF_TOKEN[-4:], len(HF_TOKEN)))''')

# =====================================================================================
md("cell-03", """## 2 · Check the GPU, driver and disk

A compute capability below 8.0, a driver older than 525, or less than ~70 GB free on `/workspace` will stop you later — better to see it now.""")

code("cell-04", '''sh("nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version "
   "--format=csv || echo 'NO GPU VISIBLE'", cwd="/")
sh("echo; df -h /workspace / 2>/dev/null | sed -n '1,5p'; echo; uname -m", cwd="/")

# a quick sanity read of the numbers above
out = subprocess.run(["bash", "-c", "nvidia-smi --query-gpu=compute_cap,memory.total "
                      "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
if out.strip():
    cap, mem = [x.strip() for x in out.strip().splitlines()[0].split(",")]
    gib = float(mem) / 1024
    print("\\ncompute capability %s, %.0f GiB of VRAM" % (cap, gib))
    if float(cap) < 8.0:
        print("!! this pipeline needs Ampere or newer (compute capability >= 8.0)")
    elif gib < 20:
        print("!! under 20 GiB of VRAM - expect CUDA OOM with the int4 trunk + STT + LLM")
    else:
        print("ok")''')

# =====================================================================================
md("cell-05", """## 3 · System libraries

`libsndfile1` for soundfile, `ffmpeg` for torchaudio/librosa, `libgl1` + `libglib2.0-0` for OpenCV, `util-linux` for the watchdog's `flock`. Already present in most RunPod images, in which case this is a no-op.""")

code("cell-06", '''sh("export DEBIAN_FRONTEND=noninteractive && "
   "apt-get update -qq && "
   "apt-get install -y -qq --no-install-recommends "
   "  git curl ca-certificates bzip2 unzip util-linux "
   "  ffmpeg libsndfile1 libgl1 libglib2.0-0", cwd="/")''')

# =====================================================================================
md("cell-07", """## 4 · Clone the tool-calling repo

The fork already contains the complete web-search pipeline (`src/niloys/websearch/`) and the
x86_64 scripts, so **no `%%writefile` cells are needed** — the repo's own files are used as-is.
Re-running this cell just pulls the latest commit.""")

code("cell-08", '''os.makedirs(os.path.dirname(REPO_DIR), exist_ok=True)
if os.path.isdir(REPO_DIR + "/.git"):
    sh("git pull --ff-only", cwd=REPO_DIR)
else:
    sh(f"git clone {REPO_URL} {REPO_DIR}" + (f" -b {BRANCH}" if BRANCH else ""),
       cwd=os.path.dirname(REPO_DIR))
sh("git log -1 --oneline", cwd=REPO_DIR)

# the scripts must be executable: git does not always preserve the execute bit (e.g. when
# the repo was uploaded from Windows or the GitHub web UI), and ./script.sh then fails with
# "Permission denied". run_watchdog_x86.sh also calls ./run_x86.sh internally, so fix them all.
sh("chmod +x *.sh && ls -l *.sh", cwd=REPO_DIR)

# sanity: the web-search package and the x86 scripts must be in the clone
need = ["src/niloys/websearch", "setup_env_x86.sh", "prepare_x86.sh", "run_x86.sh",
        "tools/test_websearch.py"]
missing = [p for p in need if not os.path.exists(os.path.join(REPO_DIR, p))]
if missing:
    raise RuntimeError("the clone is missing %s - is %s the tool-calling fork?" % (missing, REPO_URL))
print("\\nweb-search package + x86 scripts present in the clone")''')

# =====================================================================================
md("cell-09", """## 5 · Build the Python environment  *(~8–20 min, once)*

`setup_env_x86.sh` builds a Python 3.10 env in `./env`, then torch (CUDA 12.8), the pinned
`requirements.txt`, and `sphn` + TensorRT + **`transformers==4.37.2`** from
`requirements-x86-extra.txt` (the web-search LLM needs the qwen2 architecture; the server's only
other transformers use is Wav2Vec2, whose API is unchanged). The last line must be `ENV READY`.""")

code("cell-10", '''t0 = time.time()
sh("./setup_env_x86.sh")
print("\\n--- setup_env_x86.sh took %.1f min ---" % ((time.time() - t0) / 60))''')

# =====================================================================================
md("cell-11", """### Sync the pip extras

Cheap and idempotent (a few seconds once the env exists). Run it if you built `./env` from an
older copy of this notebook or an older clone — it pulls in `hf_transfer` (which RunPod's images
*require* for any Hugging Face download) and upgrades `transformers` for the web-search LLM.""")

code("cell-12", '''sh("./env/bin/python -m pip install -r requirements-x86-extra.txt")
sh("""./env/bin/python -c 'import sphn, tensorrt, hf_transfer, huggingface_hub, transformers; """
   """print("sphn/tensorrt/hf_transfer importable | tensorrt", tensorrt.__version__, """
   """"| huggingface_hub", huggingface_hub.__version__, "| transformers", transformers.__version__)'""")''')

# =====================================================================================
md("cell-13", """## 6 · Download the weights and build the TensorRT engine  *(~15–40 min, ~25 GB)*

1. environment check
2. `nvidia/personaplex-7b-v1` — bf16 trunk, Mimi codec, tokenizer, voices (**gated**, ~16 GB)
3. the Robert_5 one-pass depformer + GAN Mimi decoder (~2.9 GB, public, sha256-checked)
4. avatar weights, the Robert_5 voice and the decoder ONNX (~250 MB, public, sha256-checked)
5. **the web-search models** — `kyutai/stt-1b-en_fr` + `Qwen/Qwen2.5-1.5B-Instruct` (~5.7 GB,
   public; runs because `PREPARE_WEBSEARCH=1` was set in cell 1 when a Tavily key is present)
6. the GPU capability probe
7. the TensorRT engine for the audio decoder, built for **this** GPU

Step 2 fails if the token is wrong or the account has not accepted the model licence.
Safe to re-run; it skips what is already there. The last line must be `READY`.""")

code("cell-14", '''t0 = time.time()
sh("./prepare_x86.sh " + HF_TOKEN)
print("\\n--- prepare_x86.sh took %.1f min ---" % ((time.time() - t0) / 60))''')

# =====================================================================================
md("cell-15", """### What the probe decided

`TRY19_INT4_G=32` means the 7B trunk is quantised to int4 at load — what the Orin ships, ~14 GB
of VRAM. `TRY19_INT4_G=0` means ATen's int4 tinygemm kernels are not usable on this card, so the
trunk stays bf16: correct but slower, and it needs ~26 GB of VRAM (plus ~6 GB for the STT + LLM).""")

code("cell-16", '''sh("cat .gpu_probe.env; echo; "
   "ls -lL weights/seanet_w12_fp32.engine 2>/dev/null || echo 'no TensorRT engine "
   "(the PyTorch SEANet decoder will be used - a few ms slower, same audio)'", check=False)''')

# =====================================================================================
md("cell-17", """## 7 · Start the server

It runs detached with `setsid`, so it keeps serving while you use the rest of the notebook and
survives a kernel restart. Output goes to `run.log`. The web-search models load in the
background — `/health` shows `websearch.ready` once they are up.""")

code("cell-18", '''sh("pkill -f run_watchdog_x86.sh; pkill -f imtalker_try19_student_live.py; sleep 2", check=False)
sh("PORT=%d VOICE=%s TEXT_PROMPT=%s TAVILY_API_KEY=%s WEBSEARCH_LLM_MODEL=%s nohup setsid ./run_x86.sh > run.log 2>&1 < /dev/null & "
   "sleep 3; echo started:; pgrep -af imtalker_try19_student_live.py | head -3"
   % (PORT, VOICE, json.dumps(TEXT_PROMPT), json.dumps(TAVILY_API_KEY), json.dumps(WEBSEARCH_LLM_MODEL)))''')

# =====================================================================================
md("cell-19", """### Wait until it has loaded

Loading the 7B trunk, quantising it, capturing the CUDA graphs and compiling the Triton kernels
takes a few minutes the first time (the kernel cache in `.cache/` makes later starts quicker).
This cell polls `/health` until `loaded: true`, then prints the web-search status.""")

code("cell-20", '''def wait_loaded(port, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=5) as r:
                h = json.load(r)
            if h.get("loaded"):
                print("\\nloaded after %.0fs: %s" % (time.time() - t0, h))
                return True
        except Exception:
            pass
        if subprocess.run(["pgrep", "-f", "imtalker_try19_student_live.py"],
                          capture_output=True).returncode != 0:
            print("\\n!! the server process is gone - last 40 lines of run.log:")
            sh("tail -40 run.log", check=False)
            return False
        tail = subprocess.run(["bash", "-c", "tail -1 run.log"], capture_output=True,
                              text=True, cwd=REPO_DIR).stdout.strip()
        print("  ... %5.0fs  %s" % (time.time() - t0, tail[:110]), flush=True)
        time.sleep(15)
    print("\\n!! timed out")
    sh("tail -40 run.log", check=False)
    return False

ok = wait_loaded(PORT)
print()
# the feature lines that must be in the log (int4 / trt_seanet are absent if disabled above)
sh("grep -aE 'x86:|int4 G=|GATING|TRY19_PD hook|fine-tune hook|captured AT INIT|trt_seanet|"
   "WEBSEARCH|Uvicorn running' run.log || echo '(none found yet)'", check=False)''')

# =====================================================================================
md("cell-21", """## 8 · Open it in Chrome

**The microphone only works on `https://` or `http://localhost`.** Plain `http://<pod-ip>:8991`
loads the page, but Chrome refuses the microphone on it. Two ways to get HTTPS:

### Option A — the RunPod HTTP proxy  *(use this if 8991 is in the pod's HTTP ports)*""")

code("cell-22", '''pod = os.environ.get("RUNPOD_POD_ID", "")
if pod:
    print("  https://%s-%d.proxy.runpod.net/\\n" % (pod, PORT))
    print("If that 502s, port %d is not in this pod's HTTP ports - either add it (a template"
    "change needs the pod recreated) or use the Cloudflare tunnel below.")
else:
    print("RUNPOD_POD_ID is not set - this is not a RunPod pod, or it is an older image.")
    print("Use the Cloudflare tunnel below, or an SSH tunnel:")
    print("  ssh -L %d:localhost:%d root@<pod-host> -p <ssh-port>   then http://localhost:%d/"
          % (PORT, PORT, PORT))''')

# =====================================================================================
md("cell-23", """### Option B — Cloudflare quick tunnel  *(no pod configuration, no account)*

Downloads `cloudflared` and prints a `https://<random>.trycloudflare.com` URL that proxies to the
server, websockets included. Leave the tunnel running for as long as you want the URL; re-run the
cell to get a fresh one.""")

code("cell-24", '''import re
os.makedirs(".bin", exist_ok=True)
CF = os.path.abspath(".bin/cloudflared")
if not os.path.exists(CF):
    print("downloading cloudflared ...")
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        CF)
    os.chmod(CF, 0o755)

sh("pkill -f 'cloudflared tunnel'; sleep 1", check=False)
tun = subprocess.Popen([CF, "tunnel", "--no-autoupdate", "--url", "http://localhost:%d" % PORT],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
public, t0 = None, time.time()
while time.time() - t0 < 90:
    line = tun.stdout.readline()
    if not line:
        break
    m = re.search(r"https://[-\\w.]+\\.trycloudflare\\.com", line)
    if m:
        public = m.group(0)
        break
if public:
    print("\\n" + "=" * 72)
    print("  OPEN THIS IN CHROME:   %s/" % public)
    print("=" * 72)
    print("\\nClick 'Start conversation', allow the microphone, and speak.")
else:
    print("could not get a tunnel URL; last output:")
    print(tun.stdout.read()[-2000:] if tun.stdout else "")''')

# =====================================================================================
md("cell-25", """## 9 · Headless end-to-end check (no browser, no microphone)

Streams 20 s of recorded speech into the server, then silence, and reports what came back.
`PASS` = it heard the question, answered with speech and streamed video above 15 fps. The median
step time must stay under **80 ms** — that is the whole budget for one 80 ms audio frame.""")

code("cell-26", '''sh("$(cat .python_path) tools/test_session.py %d 40" % PORT, check=False)''')

# =====================================================================================
md("cell-27", """## 10 · Web-search check

1. transcribes the recorded test question with the STT model (no server needed),
2. streams it into the running server and prints the `[WEBSEARCH]` pipeline trace: the STT
   transcript, the LLM decision, the Tavily results, the answer, the injection and the model's
   spoken reply. PASS = the plumbing worked (NOSEARCH is a valid outcome too — it just means the
   recorded question needed no web search).""")

code("cell-28", '''# 1) transcribe the recorded test question with the STT model (no server needed)
sh("$(cat .python_path) tools/test_websearch.py --stt", check=False)''')

code("cell-29", '''# 2) stream it into the running server and print the [WEBSEARCH] pipeline trace
if TAVILY_API_KEY:
    sh("$(cat .python_path) tools/test_websearch.py --e2e %d --logfile run.log" % PORT, check=False)
else:
    print("TAVILY_API_KEY not set in cell 1 - skipped the live search test")''')

# =====================================================================================
md("cell-30", """### Watch a live search turn

Ask the avatar something current in the browser — *"what is the bitcoin price today?"*, *"who won
the game last night?"*, *"what is the weather in Tokyo?"* — then run the cell below. You get the
whole timestamped trace of the turn:

```
[WEBSEARCH][12:34:56.789] [STT] question: 'what is the bitcoin price today?'
[WEBSEARCH][12:34:57.012] [LLM-DECISION] SEARCH (234 ms) q='...' raw='SEARCH\\nbitcoin price today'
[WEBSEARCH][12:34:58.345] [SEARCH] 3 results (1102 ms) query='bitcoin price today'
[WEBSEARCH][12:34:58.345] [SEARCH]   [1] Bitcoin price today (https://...)
[WEBSEARCH][12:34:58.345] [SEARCH]       Bitcoin is trading at about ...
[WEBSEARCH][12:34:59.678] [LLM-SUMMARY] (1333 ms) answer: 'Bitcoin is ...'
[WEBSEARCH][12:35:00.901] [INJECT] queued: '<system> Web search result ...'
[WEBSEARCH][12:35:01.234] [INJECT] fed 72/72 tokens in 890 ms
[WEBSEARCH][12:35:08.456] [REPLY] model said: 'Sure - bitcoin is currently ...'
```""")

code("cell-31", '''sh("grep -a '\\\\[WEBSEARCH\\\\]' run.log | tail -45", check=False)''')

# =====================================================================================
md("cell-32", """## 11 · Day-to-day: logs, restart, stop""")

code("cell-33", '''sh("tail -60 run.log", check=False)''')

code("cell-34", '''# step times and anything worth knowing about
sh("grep -aE 'step time|over budget|WARNING|Error|Traceback' run.log | tail -40", check=False)''')

code("cell-35", '''# just the web-search pipeline trace
sh("grep -a '\\\\[WEBSEARCH\\\\]' run.log | tail -40", check=False)''')

code("cell-36", '''# restart under the watchdog instead (auto-restarts on a crash; logs to run.log + crashlogs/)
sh("pkill -f imtalker_try19_student_live.py; sleep 2; "
   "PORT=%d TAVILY_API_KEY=%s WEBSEARCH_LLM_MODEL=%s nohup setsid ./run_watchdog_x86.sh > /dev/null 2>&1 < /dev/null & "
   "sleep 3; pgrep -af run_watchdog_x86.sh"
   % (PORT, json.dumps(TAVILY_API_KEY), json.dumps(WEBSEARCH_LLM_MODEL)), check=False)''')

code("cell-37", '''# stop everything
sh("pkill -f run_watchdog_x86.sh; pkill -f imtalker_try19_student_live.py; "
   "pkill -f 'cloudflared tunnel'; sleep 2; "
   "nvidia-smi --query-gpu=memory.used --format=csv,noheader", check=False)''')

# =====================================================================================
md("cell-38", """## 12 · Tuning and troubleshooting

### Knobs — set them in cell 1 (or export them and re-run `./run_x86.sh`)

| variable | default | effect |
|---|---|---|
| `PORT` | 8991 | server port (8998 is refused by the server) |
| `VOICE` | `Robert_5.pt` | voice prompt from `weights/voices/`. The depformer and the decoder were trained for Robert_5 — other voices sound clearly worse |
| `TEXT_PROMPT` | Robert, RB Labs | persona / system prompt |
| `PREBUFFER` | 3 | pre-roll chunks of 80 ms. Raise it if audio stutters on a slow network |
| `TRY19_INT4_G` | from the probe | `32` = int4 trunk (~14 GB VRAM), `0` = bf16 trunk (~26 GB, slower) |
| `TRY19_MIMI_GRAPH` | 1 | Mimi encoder CUDA graph. Set `0` if graph capture misbehaves on your card |
| `TRY19_RENDER_FP16` | 1 | fp16 renderer. `0` for fp32 (slower, 64 dB difference) |
| `GAIN` | 1.2 | motion gain of the avatar |
| `TAVILY_API_KEY` | _(unset)_ | set it to enable web search: STT + local Qwen2.5-1.5B LLM + Tavily top-3, injected into PersonaPlex. Unset = classic pipeline |
| `WEBSEARCH_LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | local HF LLM for search routing + answers (use `Qwen/Qwen2.5-0.5B-Instruct` on tight VRAM) |
| `WEBSEARCH_DRY_RUN` | _(unset)_ | `1` = log the would-be injection without touching PersonaPlex |
| `WEBSEARCH_STT_INT4` | _(unset)_ | `1` = int4-quantize the STT transformer (Orin / tight budgets) |

Example: `PORT=8080 TRY19_INT4_G=0 ./run_x86.sh 2>&1 | tee run.log`

### Troubleshooting

| symptom | fix |
|---|---|
| microphone blocked, no "allow" prompt | the page must be on `https://` or `http://localhost`. Use the Cloudflare tunnel or an SSH tunnel |
| the proxy URL 502s | port `8991` is not in the pod's HTTP ports, or the server has not finished loading — check `/health` first |
| `int4 tinygemm : UNAVAILABLE` in the probe | normal on some architectures. The trunk falls back to bf16 and then needs ~26 GB of VRAM: use a bigger GPU, or accept the slower path |
| `CUDA out of memory` while loading | something else holds VRAM (`nvidia-smi`), or the card is too small for the plan the probe chose. With web search on, budget ~20 GB total (int4 trunk + STT + LLM) |
| step 2 of `prepare_x86.sh` fails | the token is wrong, or the HF account has not accepted the `nvidia/personaplex-7b-v1` licence |
| `Fast download using 'hf_transfer' is enabled ... but 'hf_transfer' package is not available` | the image exports `HF_HUB_ENABLE_HF_TRANSFER=1`. Run the "sync the pip extras" cell in section 5, then re-run section 6 |
| no TensorRT engine | see `weights/trtexec_build.log`. Not fatal — Mimi's PyTorch SEANet decoder is used instead, a few ms slower per step with identical audio |
| browser connects then disconnects at once | `sphn` must be the **pip** 0.2.x build. `src/pylibs` holds an aarch64 `sphn` and must stay off `PYTHONPATH` on x86 — `run_x86.sh` already keeps it off |
| choppy audio, step time over 80 ms | `env/bin/pip uninstall uvloop`; make sure nothing else is using the GPU; run one server per GPU |
| `ImportError: libGL.so.1` | re-run cell 3 |
| disk full mid-download | the volume is under ~70 GB. `du -sh hf weights env .cache` shows where it went |
| `websearch.ready: false` in `/health` | the models are still downloading (first start), or the load failed — `grep WEBSEARCH run.log` shows why. The classic pipeline keeps working either way |
| no `[WEBSEARCH]` lines in `run.log` | `TAVILY_API_KEY` was not set when the server started, or the question was classified NOSEARCH. Ask something clearly current (price, news, weather) |
| searches but no injection | the answer arrived after the 3.5 s deadline (logged), or the user barged in. Both are intentional fallbacks |
| a question is misheard (e.g. "what is bitcoin?" came out as "parties Bitcoin") | fixed in v2: an 800 ms pre-roll now feeds the soft onset to the STT. If a question is still clipped, say the first word a touch louder |
| after a search the answer starts then stops, or the lips move ~2-3 s before the voice | fixed in v2: the mic audio that piled up during the search pause is now drained on resume, so no catch-up burst desyncs A/V. If it ever recurs, `grep EVENT-BACKLOG run.log` should NOT show a stuck high depth |
| a question right after the avatar answers gets no `[STT] question` line | fixed in v2 (the echo guard now tracks playback, so the avatar's own voice is never transcribed). If it recurs, `grep "DIAG empty turn" run.log` shows why |
| a stable question ("What is Bitcoin?") triggers a web search | fixed in v3: a deterministic stable-question guard now NOSEARCHes definition/explanation/opinion questions, and the classifier uses few-shot examples. Only current-info questions (price, today, news, ...) search |
| the avatar says "let me know if you have any other questions" instead of answering | fixed in v3: the injection no longer states the answer as a finished sentence (which made the model think it had already spoken), and the session-start PAD-silence that triggered the greeting was removed. Knowledge questions are now answered directly with no injection at all |
| several questions in one message all get searched | fixed in v3: the utterance is split into individual questions, each routed independently; only the current-information ones are searched and one combined injection answers all of them in order |

### Known behaviour (same as on the Orin)

* With very long continuous speech the avatar video can fall a few seconds behind; normal
  back-and-forth conversation stays ~0.1–0.15 s behind.
* The model's context window is 4 minutes, so very long sessions drift after ~4–6 minutes.
  Reloading the page starts a fresh session.
* A search turn adds a few seconds of "thinking" pause before the answer — that is the
  classification + search + answer + injection time, and it is intentional.
* Everything lives under `/workspace/speech2face_toolcalling`. On a pod restart with the same
  volume, skip cells 5 and 6 and go straight to cell 7.""")

# =====================================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print("wrote", OUT, "with", len(cells), "cells")

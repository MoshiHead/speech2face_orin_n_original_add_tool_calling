# Running on RunPod (or any x86_64 NVIDIA GPU)

The Orin path (`setup_env.sh` -> `prepare.sh` -> `run.sh`) is unchanged. This page covers the
x86_64 path, which runs the **same model configuration** with the Jetson-only assumptions removed.

```bash
./setup_env_x86.sh                       # once, ~8-20 min: Python 3.10 env in ./env (Miniforge)
./prepare_x86.sh hf_YOUR_TOKEN           # once: ~19 GB of weights, GPU probe, TensorRT engine
export PORT=8991
./run_x86.sh 2>&1 | tee run.log          # or ./run_watchdog_x86.sh & for auto-restart
```

`runpod_speech2face.ipynb` does all of the above from a notebook and also sets up an HTTPS URL.

## What differs from the Jetson scripts

| | Jetson (`*.sh`) | x86_64 (`*_x86.sh`) |
|---|---|---|
| torch | NVIDIA Jetson index, CUDA 12.6 | `download.pytorch.org/whl/cu128`, torch 2.8.0+cu128 (Ampere / Ada / Blackwell) |
| `sphn` | bundled aarch64 wheel in `src/pylibs` | `sphn==0.2.1` from PyPI (`src/pylibs` is kept **off** `PYTHONPATH`) |
| TensorRT | JetPack, `trtexec` | `tensorrt-cu12` from pip; the engine is built with `tools/build_trt_engine.py` (no `trtexec` in the pip wheels) |
| TensorRT engine | required | **optional** -- without it Mimi's PyTorch SEANet decoder is used (a few ms slower per step, same audio) |
| int4 trunk | always on (`TRY19_INT4_G=32`) | `tools/gpu_probe.py` tries ATen's tinygemm int4 kernels on the actual GPU and writes `.gpu_probe.env`; `run_x86.sh` honours it |

Everything else -- the Robert_5 one-pass depformer, the GAN Mimi decoder, the encoder CUDA
graph, the motion student and the distilled renderer -- is byte-identical to the Orin run.

## Requirements

| what | needed |
|---|---|
| GPU | NVIDIA, compute capability >= 8.0 (Ampere or newer): A100, A6000, L40S, 4090, 5090, H100. ~14 GB of VRAM with the int4 trunk, ~26 GB if the int4 kernels are unusable on that card |
| driver | >= 525 (CUDA 12.x). `nvidia-smi` must work inside the container |
| disk | ~60 GB free on the volume you clone to (weights ~19 GB, env ~8 GB, HF cache) |
| Hugging Face | a token whose account has **accepted the licence of `nvidia/personaplex-7b-v1`** |
| network | github.com, conda-forge, PyPI, download.pytorch.org, huggingface.co |
| system packages | `ffmpeg libsndfile1 libgl1 git curl` (the RunPod PyTorch images already have most of them) |

## Microphone needs HTTPS

Chrome only grants microphone access on `https://` or `http://localhost`. On a pod:

* **RunPod HTTP proxy** -- expose the port in the pod template, then open
  `https://<POD_ID>-8991.proxy.runpod.net/`. Websockets are proxied.
* **Cloudflare quick tunnel** -- no pod configuration and no account:
  `cloudflared tunnel --url http://localhost:8991` prints a `https://<random>.trycloudflare.com` URL.
* **SSH tunnel** -- `ssh -L 8991:localhost:8991 root@<pod> -p <port>`, then `http://localhost:8991/`.

## Verify it works without a browser

```bash
until curl -s http://localhost:8991/health | grep -q '"loaded":true'; do sleep 10; done
$(cat .python_path) tools/test_session.py 8991 40
```

`PASS` means it heard the question, answered with speech and streamed video above 15 fps.
The step time printed must stay under 80 ms for real time.

The log should contain:

```
x86: trunk int4 G=32 + robert5 one-pass depformer + GAN decoder (TensorRT) voice=Robert_5.pt port=8991
[try19] int4 G=32: 128 Linears at load ...
[try19] depformer GATING int4: 192 Linears ...
[try19] TRY19_PD hook installed -> .../weights/pd_quant_onpol_robert5_plaus2.pt ...
[try19] Mimi decoder fine-tune hook installed -> .../weights/mimi_dec_gan_robert5.pt
[try19] mimi encode graph captured AT INIT: True
[trt_seanet] mimi.decoder -> TensorRT .../weights/seanet_w12_fp32.engine (window 12 steps = 480 ms)
INFO:     Uvicorn running on http://0.0.0.0:8991
```

The `int4` and `trt_seanet` lines are absent when the GPU probe disabled int4 or no engine was
built; the server still works.

## Optional: web search (STT + local LLM + Tavily)

Set `TAVILY_API_KEY` (free tier at tavily.com) before `prepare_x86.sh` / `run_x86.sh` and the
avatar gains an online-search turn:

1. `kyutai/stt-1b-en_fr` transcribes your speech in a side thread (streaming, ~2.6 GB VRAM).
2. A small **local** instruct LLM — `Qwen/Qwen2.5-1.5B-Instruct` by default, downloaded from
   Hugging Face, **no LLM API** (~3.1 GB VRAM) — decides whether the question needs current
   online information. If not, nothing else happens: the pipeline is byte-identical.
3. If yes: Tavily returns the top 3 results, the same LLM writes one short answer, and the
   answer is fed into PersonaPlex through its native text-prompt stream (the same mechanism
   as the persona system prompt), so it replies in its own voice. The last 4 conversation
   turns are kept so follow-up questions resolve correctly.

Any failure (no key, download error, search timeout, LLM error) just skips the injection —
the classic pipeline always answers. `WEBSEARCH_DRY_RUN=1 ./run_x86.sh` logs the would-be
injection without touching the model. `WEBSEARCH_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct` uses
a ~1 GB LLM for tight VRAM. The models download in `prepare_x86.sh` (when the key is set) or
lazily in the background at server start.

Notes:

* `requirements-x86-extra.txt` upgrades `transformers` to 4.37.2 (the qwen2 arch needs it;
  the server's only transformers use is Wav2Vec2, whose API is unchanged 4.30 -> 4.37).
* VRAM: +~6 GB on top of the int4 config (~20 GB total on a 24 GB card — tight but OK).
* Latency: a search turn adds roughly 2–4 s of "thinking" before the answer (classification
  + search + answer + the token-budgeted injection). Non-search turns add ~nothing.
* Test it: `$(cat .python_path) tools/test_websearch.py --stt` (transcribes the test wav),
  `--classify "question"`, `--search "query"`, `--e2e 8991 --logfile run.log`.

## Troubleshooting

| symptom | fix |
|---|---|
| `this script is for x86_64` | you are on a Jetson: use `setup_env.sh` / `prepare.sh` / `run.sh` |
| `nvidia-smi not found` | the pod was started without a GPU, or without the NVIDIA runtime |
| `int4 tinygemm : UNAVAILABLE` in the probe | normal on some architectures. The trunk then runs bf16 and needs ~26 GB of VRAM; pick a bigger GPU or accept the slower path |
| `CUDA out of memory` at load | another process is holding VRAM (`nvidia-smi`), or the card is too small for the plan the probe chose |
| `hf_transfer package is not available` | the image exports `HF_HUB_ENABLE_HF_TRANSFER=1`. `pip install -r requirements-x86-extra.txt` into `./env` (or re-run `./setup_env_x86.sh`); `prepare_x86.sh` also disables the accelerator itself when the package is missing |
| step 2 of `prepare_x86.sh` fails | token wrong, or the HF account has not accepted the `nvidia/personaplex-7b-v1` licence |
| no TensorRT engine | see `weights/trtexec_build.log`. Not fatal; the PyTorch SEANet decoder is used |
| browser connects then disconnects | `sphn` must be 0.2.x and must be the **pip** one -- `src/pylibs` must not be on `PYTHONPATH` on x86 |
| microphone blocked | the page must be on HTTPS or `localhost`; see above |
| choppy audio, step time > 80 ms | `pip uninstall uvloop`; make sure nothing else uses the GPU; one server per GPU |
| `ImportError: libGL.so.1` | `apt-get install -y libgl1` |

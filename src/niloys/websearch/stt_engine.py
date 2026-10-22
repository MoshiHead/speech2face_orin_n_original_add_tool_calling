"""Streaming speech-to-text with kyutai/stt-1b-en_fr on THIS repo's own moshi classes.

Why not the pip `moshi` package: this repo carries its own `moshi` fork (adapted for
PersonaPlex) first on PYTHONPATH, so the official package cannot be imported alongside.
Instead this module loads the STT checkpoint with the fork's LMModel/MimiModel -- the
state-dict key layout is the same lineage (the fork's _remap_state_dict_keys already
handles the one rename) -- and drives it with a small hand-written streaming loop.

STT model facts (from kyutai/stt-1b-en_fr config.json):
  Mimi 24 kHz / 12.5 Hz, 32 codebooks; transformer dim 2048, 16 layers, dep_q=0,
  all delays 0, own 8k SentencePiece tokenizer, 0.5 s text delay, greedy decoding.

The loop below reproduces the official LMGen behaviour for dep_q=0 / delays=0 models
(see kyutai's delayed-streams-modeling stt_pytorch notebook):
  step 0: forward([text_initial=8000, audio_initial=2048 x32])  -> text_1
  step t: forward([text_{t-1}, mimi_codes(frame_{t-1})])        -> text_t
"""
from __future__ import annotations

import contextlib
import json
import queue
import threading
import time

import numpy as np

FRAME_SAMPLES = 1920          # 80 ms @ 24 kHz
SAMPLE_RATE = 24000
FRAME_SEC = FRAME_SAMPLES / SAMPLE_RATE
_SKIP_TEXT_TOKENS = (0, 3)    # end-of-word marker / text padding (official notebook skips these)

# config.json keys that LMModel.__init__ (or its **kwargs -> StreamingTransformer) accepts.
_LM_CONFIG_KEYS = (
    "delays", "n_q", "dep_q", "card", "text_card", "dim", "num_heads", "hidden_scale",
    "norm", "norm_emb", "bias_proj", "depformer_dim", "depformer_dim_feedforward",
    "depformer_multi_linear", "depformer_weights_per_step", "depformer_pos_emb",
    "existing_text_padding_id", "context", "num_layers", "max_period", "gating",
    "causal", "layer_scale", "positional_embedding", "depformer_num_heads",
    "depformer_num_layers", "depformer_context", "depformer_max_period", "depformer_gating",
    "depformer_layer_scale", "dim_feedforward",
)


class KyutaiSTT:
    """Streaming STT engine: mic PCM in (feed), turn-complete questions out (callback)."""

    def __init__(self, repo: str, device: str, int4: bool = False,
                 vad_rms: float = 0.01, vad_speech_ms: int = 500, vad_silence_ms: int = 500,
                 flush_quiet_ms: int = 500, flush_max_ms: int = 1400,
                 min_question_words: int = 2, assistant_rms: float = 0.006,
                 assistant_hold_s: float = 0.5, log=print):
        import torch
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        self.log = log
        self.device = torch.device(device)
        self.vad_rms = float(vad_rms)
        self.vad_speech_ms = int(vad_speech_ms)
        self.vad_silence_ms = int(vad_silence_ms)
        self.flush_quiet_ms = int(flush_quiet_ms)
        self.flush_max_ms = int(flush_max_ms)
        self.min_question_words = int(min_question_words)
        self.assistant_rms = float(assistant_rms)
        self.assistant_hold_s = float(assistant_hold_s)

        cfg_path = hf_hub_download(repo, "config.json")
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        mimi_path = hf_hub_download(repo, cfg.get("mimi_name", "mimi-pytorch-e351c8d8@125.safetensors"))
        lm_path = hf_hub_download(repo, "model.safetensors")
        tok_path = hf_hub_download(repo, cfg.get("tokenizer_name", "tokenizer_en_fr_audio_8000.model"))

        from moshi.models import loaders
        from moshi.models.lm import LMModel

        self.mimi = loaders.get_mimi(mimi_path, self.device)
        self.mimi.set_num_codebooks(32)   # the STT consumes the full codec (get_mimi sets 8)

        lm_kwargs = {k: cfg[k] for k in _LM_CONFIG_KEYS if k in cfg}
        self.lm = LMModel(device="meta", dtype=torch.bfloat16, **lm_kwargs)
        state = loaders._remap_state_dict_keys(load_file(lm_path, device="cpu"))
        for key in state:
            state[key] = state[key].to(device=self.device, dtype=torch.bfloat16)
        res = self.lm.load_state_dict(state, strict=False, assign=True)
        missing = [k for k in res.missing_keys if not k.startswith("depformer")]
        if missing:
            raise RuntimeError(
                f"STT checkpoint missing {len(missing)} non-depformer keys: {missing[:5]}"
            )
        if res.unexpected_keys:
            self.log(f"[STT] note: {len(res.unexpected_keys)} unexpected "
                     f"checkpoint keys ignored (e.g. {res.unexpected_keys[:3]})")
        del state
        # dep_q=0: the depformer is never called -- drop it instead of materialising it.
        self.lm.depformer = None
        self.lm.depformer_text_emb = None
        self.lm.to(device=self.device, dtype=torch.bfloat16)
        self.lm.eval()
        if int4:
            try:
                from int4_linear import convert_bf16
                n = convert_bf16(self.lm.transformer, 32)
                self.log(f"[STT] int4-quantized {n} transformer linears")
            except Exception as e:  # noqa: BLE001
                self.log(f"[STT] int4 unavailable ({e!r}) -- staying bf16")
        self.tokenizer = sentencepiece.SentencePieceProcessor(tok_path)

        self.mimi.streaming_forever(1)
        self.lm.streaming_forever(1)
        self._disable_mimi_graphs()

        self._q: queue.Queue = queue.Queue(maxsize=600)   # ~48 s of 80 ms frames
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._session_active = False
        self.on_turn_complete = None      # set by the orchestrator: fn(question, turn_end_time)
        self.on_turn_closing = None       # set by the orchestrator: fn(partial_text) at flush start
        self._reset_model_state()
        self._reset_turn_state()


    # ------------------------------------------------------------------ model
    def _disable_mimi_graphs(self) -> None:
        """Force this Mimi instance to run fully eager.

        MimiModel._init_streaming_state wraps the encoder/decoder transformers in
        CUDAGraphed, which CAPTURES lazily on first use. A capture runs in 'global'
        error mode, which cannot tolerate the persona/render threads' concurrent CUDA
        work -- it crashes them ('operation not permitted when stream is capturing',
        'Offset increment outside graph capture'). The STT is a side thread and does
        not need the graph speedup, so pin its wrappers to eager. reset_streaming()
        re-creates the wrappers, so this is re-applied after every reset.
        """
        st = getattr(self.mimi, "_streaming_state", None)
        for attr in ("graphed_tr_enc", "graphed_tr_dec"):
            g = getattr(st, attr, None) if st is not None else None
            if g is not None:
                g.disable = True

    def _reset_model_state(self) -> None:
        with contextlib.suppress(Exception):
            self.mimi.reset_streaming()
        self._disable_mimi_graphs()
        with contextlib.suppress(Exception):
            self.lm.reset_streaming()
        self._prev_input = self.lm._get_initial_token()   # [1, 33, 1]

    def _reset_turn_state(self) -> None:
        self._in_turn = False
        self._flushing = False
        self._turn_text = ""
        self._turn_speech_frames = 0
        self._turn_peak_rms = 0.0
        self._silence_run = 0
        self._last_text_at = 0.0
        self._flush_start = 0.0
        self._last_user_speech_at = 0.0
        self._assistant_speaking_until = 0.0
        self._last_mic_rms = 0.0

    def _model_frame(self, frame_f32: np.ndarray) -> str:
        """One 80 ms frame through Mimi + STT LM -> newly emitted text (maybe '')."""
        import torch
        with torch.no_grad():
            chunk = torch.from_numpy(frame_f32).to(self.device, torch.float32).view(1, 1, -1)
            codes = self.mimi.encode(chunk)[:, :, :1]                     # [1, 32, 1]
            _, text_logits = self.lm.forward_codes(self._prev_input)
            token = int(text_logits[0, 0, 0].argmax().item())
            self._prev_input = torch.cat(
                [torch.full((1, 1, 1), token, device=self.device, dtype=torch.long), codes],
                dim=1,
            )
        if token in _SKIP_TEXT_TOKENS:
            return ""
        with contextlib.suppress(Exception):
            return self.tokenizer.id_to_piece(token).replace("▁", " ")
        return ""

    # ------------------------------------------------------------------ feeding
    def feed(self, pcm_f32: np.ndarray) -> None:
        """Mic tap: 24 kHz float32 PCM of any length. Non-blocking, drop-oldest."""
        if not self._session_active or self._stop.is_set():
            return
        try:
            self._q.put_nowait(pcm_f32)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._q.get_nowait()
            with contextlib.suppress(queue.Full):
                self._q.put_nowait(pcm_f32)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._worker, name="websearch-stt", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def session_start(self) -> None:
        with self._lock:
            self._reset_turn_state()
            self._reset_model_state()
            while True:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._session_active = True

    def session_end(self) -> None:
        with self._lock:
            self._session_active = False
            self._reset_turn_state()

    # ------------------------------------------------------------------ echo guard / barge-in
    def note_assistant_rms(self, rms: float) -> None:
        """The avatar is speaking -> do not let its own voice start a user turn."""
        if rms >= self.assistant_rms:
            self._assistant_speaking_until = time.perf_counter() + self.assistant_hold_s

    def user_speaking_recently(self, window_s: float = 0.25) -> bool:
        return (time.perf_counter() - self._last_user_speech_at) < window_s



    # ------------------------------------------------------------------ worker
    def _worker(self) -> None:
        buf = np.zeros(0, dtype=np.float32)
        while not self._stop.is_set():
            try:
                pcm = self._q.get(timeout=0.1)
                buf = np.concatenate([buf, pcm])
            except queue.Empty:
                pass
            while buf.shape[0] >= FRAME_SAMPLES:
                frame, buf = buf[:FRAME_SAMPLES], buf[FRAME_SAMPLES:]
                self._process_frame(frame)
            self._check_flush_timeout()

    def _process_frame(self, frame: np.ndarray) -> None:
        now = time.perf_counter()
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if frame.size else 0.0
        speaking = rms >= self.vad_rms
        with self._lock:
            if not self._session_active:
                return
            self._last_mic_rms = rms
            if speaking:
                self._last_user_speech_at = now
            if not self._in_turn:
                # echo guard: never OPEN a turn while the avatar's voice may be in the mic
                if speaking and now >= self._assistant_speaking_until:
                    # fresh codec/LM context per utterance (the official per-file usage),
                    # so idle gaps cannot leak stale convolution state into the question
                    self._reset_model_state()
                    self._in_turn = True
                    self._turn_text = ""
                    self._turn_speech_frames = 0
                    self._turn_peak_rms = rms
                    self._silence_run = 0
                    self._flushing = False
                    self.log(f"[STT] turn open (rms={rms:.4f})")
                else:
                    return
            if speaking:
                self._silence_run = 0
                self._turn_speech_frames += 1
                self._turn_peak_rms = max(self._turn_peak_rms, rms)
            else:
                self._silence_run += 1
            try:
                piece = self._model_frame(frame)
            except Exception as e:  # noqa: BLE001
                self.log(f"[STT] model frame failed ({e!r}); skipping frame")
                piece = ""
            if piece:
                self._turn_text += piece
                self._last_text_at = now
            if (not self._flushing and not speaking
                    and self._silence_run * FRAME_SEC * 1000.0 >= self.vad_silence_ms
                    and self._turn_speech_frames * FRAME_SEC * 1000.0 >= self.vad_speech_ms):
                self._flushing = True
                self._flush_start = now
                # early hook: the turn is closing -- let the orchestrator gate PersonaPlex
                # BEFORE the model starts answering, using the partial transcript
                if self.on_turn_closing is not None:
                    try:
                        self.on_turn_closing(" ".join(self._turn_text.split()))
                    except Exception as e:  # noqa: BLE001
                        self.log(f"[STT] turn-closing callback failed ({e!r})")
            if self._flushing:
                quiet_ms = (now - self._last_text_at) * 1000.0
                if quiet_ms >= self.flush_quiet_ms or (now - self._flush_start) * 1000.0 >= self.flush_max_ms:
                    self._fire_turn(now)

    def _check_flush_timeout(self) -> None:
        with self._lock:
            if self._session_active and self._flushing:
                now = time.perf_counter()
                if (now - self._flush_start) * 1000.0 >= self.flush_max_ms:
                    self._fire_turn(now)

    def _fire_turn(self, now: float) -> None:
        question = " ".join(self._turn_text.split())
        peak = self._turn_peak_rms
        self._in_turn = False
        self._flushing = False
        self._turn_text = ""
        self._turn_speech_frames = 0
        self._silence_run = 0
        if len(question.split()) >= self.min_question_words and self.on_turn_complete is not None:
            self.log(f"[STT] question: {question!r} (peak_rms={peak:.4f})")
            try:
                self.on_turn_complete(question, now)
            except Exception as e:  # noqa: BLE001
                self.log(f"[STT] turn callback failed ({e!r})")
        elif question:
            self.log(f"[STT] ignoring short utterance: {question!r} (peak_rms={peak:.4f})")
        else:
            self.log(f"[STT] turn closed with no transcript (peak_rms={peak:.4f}) -- "
                     f"if this was speech, the STT may not be transcribing")

    # ------------------------------------------------------------------ offline (tests)
    def transcribe_offline(self, pcm_f32: np.ndarray) -> str:
        """Feed a whole utterance (24 kHz float32) and return the transcript."""
        self._reset_model_state()
        pad = np.zeros(int(1.5 * SAMPLE_RATE), dtype=np.float32)   # flush the 0.5 s text delay
        pcm = np.concatenate([np.asarray(pcm_f32, dtype=np.float32), pad])
        text = ""
        for i in range(0, pcm.shape[0] - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            text += self._model_frame(pcm[i:i + FRAME_SAMPLES])
        self._reset_model_state()
        return " ".join(text.split())

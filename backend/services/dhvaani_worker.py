"""DhVaani-0.5 text-to-speech worker (ARTPARK-IISc): zero-shot voice cloning in Indian languages.

Runs in its own virtualenv (backend/.venv-dhvaani: its torch / numpy pins differ from the other engines), started and
owned by the API server through services/dhvaani_clone.py. Only listens on 127.0.0.1.

    POST /generate  {"text", "ref_audio_path": "<wav path>", "ref_text": "<exact transcript of that wav>",
                     "language"?: "mr", "num_step"?: 16, "speed"?: 1.0, "seed"?: 666}
                    -> audio/wav (mono float32, 24 kHz)
    GET  /health    -> {"ok": true, "model": "...", "device": "mps", "languages": [...], "sample_rate": 24000}

    python dhvaani_worker.py --port 8768                 serve (the weights must already be downloaded)
    python dhvaani_worker.py --download                  fetch the gated weights (needs HF_TOKEN) and the vocoder, exit
    python dhvaani_worker.py --selftest                  speak one Hindi sentence and exit non-zero if it is not speech

The model is ZipVoice fine-tuned on Indian-script text: it conditions on a 3-10 s reference clip AND that clip's exact
transcript, and the script of the text decides the language (there is no language-id input; `language` is only checked
against the supported list). Characters outside its 1058-symbol vocabulary are silently dropped.

One generation at a time, always on the same thread (Apple's GPU is not safe to drive from several threads). Device:
CUDA when present, else Apple's GPU (MPS), else the CPU. The model stays loaded until the API server's model manager
stops this process after its idle timeout; the worker also exits when the API server dies (parent_watch.py).
"""
import argparse
import concurrent.futures
import json
import os
import re
import sys
import tempfile
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_MODEL = "ARTPARK-IISc/DhVaani-0.5"
VOCODER_REPO = "charactr/vocos-mel-24khz"
# What the snapshot download skips: the other sample clips (the Hindi one is the self-test's reference) and git metadata.
DOWNLOAD_IGNORE = ["samples/malayalam.wav", "samples/malayalam_codeswitch.wav", "samples/tamil.wav", ".gitattributes"]
REQUIRED_FILES = ("model.safetensors", "model.json", "tokens.txt", "dhvaani.py", "_backend")
SAMPLE_RATE = 24000
MAX_TEXT_CHARS = 1500
DEFAULT_STEPS = 8   # 8: about twice as fast as 16 with the same read-back accuracy (docs/RESULTS.md); DHVAANI_STEPS=16 restores the old setting
MIN_STEPS, MAX_STEPS = 4, 32
# The languages the model covers well or fairly (its model card). Sanskrit, Dogri, Kashmiri and Rajasthani are the weak
# ones (about a tenth of the training data); they are left out of the list the API routes to.
LANGUAGES = ("as", "bn", "brx", "bho", "gu", "hi", "hne", "kn", "kok", "mag", "mai", "ml", "mni", "mr", "ne", "or",
             "pa", "sat", "sd", "ta", "te", "ur", "en")
SELFTEST_TEXT = "नमस्ते, आपकी अपॉइंटमेंट कल सुबह दस बजे के लिए पक्की कर दी गई है।"
SELFTEST_REF_TEXT = "नमस्ते, कैसे हैं आप। आज मौसम बहुत अच्छा है।"


def clamp(value, lo, hi, default):
    try:
        return min(hi, max(lo, float(value)))
    except (TypeError, ValueError):
        return default


def clean_text(text) -> str:
    """Spaces collapsed, control characters removed. The model is character level; nothing else is rewritten."""
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", str(text or ""))
    return re.sub(r"[ \t]+", " ", text).strip()


def default_hf_home() -> str:
    return os.getenv("HF_HOME") or os.path.join(os.getenv("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
                                                "huggingface")


def pick_device(torch, override: str = "auto") -> str:
    if override and override != "auto":
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _watch_parent() -> None:
    """Exit if the API server that started this worker dies (see parent_watch.py)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parent_watch import watch_parent
    watch_parent()


# ------------------------------------------------------------------------------------------------ download

def download(model: str) -> int:
    """Fetch the gated weights and the vocoder, so serving can run fully offline. Exit codes: 0 ok, 2 no access, 1 other."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.setdefault("HF_HOME", default_hf_home())
    token = os.getenv("HF_TOKEN") or None
    try:
        path = snapshot_download(model, token=token, ignore_patterns=DOWNLOAD_IGNORE)
    except (GatedRepoError, RepositoryNotFoundError):
        print(f"[dhvaani] Access to {model} is not granted for this Hugging Face token. The repository is gated: "
              f"ask ARTPARK-IISc for access at https://huggingface.co/{model}, then run this again with HF_TOKEN set.",
              file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"[dhvaani] download failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"[dhvaani] weights ready: {path}", flush=True)
    try:
        snapshot_download(VOCODER_REPO, token=token)
    except Exception as e:  # noqa: BLE001
        print(f"[dhvaani] vocoder download failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print("[dhvaani] vocoder ready", flush=True)
    return 0


def snapshot_dir(model: str) -> str:
    """The cached snapshot folder, checked for the files serving needs."""
    from huggingface_hub import try_to_load_from_cache

    weights = try_to_load_from_cache(model, "model.json")
    if not isinstance(weights, str):
        raise SystemExit(f"[dhvaani] {model} is not downloaded. Run: bash backend/scripts/setup_dhvaani.sh")
    snapshot = os.path.dirname(weights)
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(snapshot, f))]
    if missing:
        raise SystemExit(f"[dhvaani] the downloaded copy of {model} is incomplete (missing {', '.join(missing)}). "
                         "Run: bash backend/scripts/setup_dhvaani.sh")
    return snapshot


# ------------------------------------------------------------------------------------------------ the model

class Engine:
    """The loaded model. Everything (load and every generation) runs on one thread, one request at a time."""

    def __init__(self, model_id: str, device: str = "auto"):
        warnings.filterwarnings("ignore")
        # The model's own module sets HF_HOME to a folder inside the snapshot unless it is already set, which would
        # hide the vocoder from the real cache; pin it to the real one first.
        os.environ.setdefault("HF_HOME", default_hf_home())
        os.environ["HF_HUB_OFFLINE"] = "1"  # never download at request time; --download does that once
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # ops Apple's GPU lacks run on the CPU instead of failing
        import torch
        self.torch = torch
        self._soundfile_for_loading()
        snapshot = snapshot_dir(model_id)
        # The model's wrapper looks for its bundled code (_backend/) next to its resolved file, but in the Hugging Face
        # cache the file is a symlink into blobs/, so put both folders on the path ourselves.
        sys.path.insert(0, os.path.join(snapshot, "_backend"))
        sys.path.insert(0, snapshot)
        from dhvaani import DhVaani  # the model's own wrapper, from the snapshot
        self.device = pick_device(torch, device)
        self.model_dir = snapshot
        self.tts = DhVaani(model_dir=snapshot, device=self.device)
        self.sr = int(getattr(self.tts, "sampling_rate", SAMPLE_RATE))
        self.languages = list(LANGUAGES)

    def _soundfile_for_loading(self) -> None:
        """torchaudio 2.11 reads audio through torchcodec (which needs ffmpeg). When that is broken, read WAV files with
        soundfile instead, so a missing ffmpeg does not stop the worker."""
        import torchaudio
        real_load = torchaudio.load

        def load(path, *args, **kwargs):
            try:
                return real_load(path, *args, **kwargs)
            except Exception:  # noqa: BLE001
                import soundfile as sf
                data, rate = sf.read(str(path), dtype="float32", always_2d=True)
                return self.torch.from_numpy(data.T.copy()), rate

        torchaudio.load = load

    def generate(self, text: str, ref_audio_path: str, ref_text: str, language=None, num_step=DEFAULT_STEPS,
                 speed=1.0, seed=666):
        import numpy as np
        import soundfile as sf
        text = clean_text(text)
        ref_text = clean_text(ref_text)
        if not text:
            raise ValueError("there is no text to speak")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text is longer than {MAX_TEXT_CHARS} characters")
        if not ref_text:
            raise ValueError("ref_text is required: DhVaani needs the exact transcript of the reference clip")
        if not ref_audio_path or not os.path.isfile(ref_audio_path):
            raise ValueError("ref_audio_path must be a readable wav file")
        if language and str(language).lower().split("-")[0] not in self.languages:
            raise ValueError(f"unsupported language '{language}'")
        fd, out_path = tempfile.mkstemp(prefix="dhvaani_out_", suffix=".wav")
        os.close(fd)
        try:
            self.tts.synthesize(text, ref_audio_path, ref_text, out_path, num_step=int(num_step), speed=float(speed),
                                seed=int(seed))
            audio, rate = sf.read(out_path, dtype="float32")
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass
            if self.device == "mps":
                try:
                    self.torch.mps.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("the model returned no usable audio")
        return audio, int(rate)


ENGINE = None
MODEL_NAME = DEFAULT_MODEL
EXECUTOR = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, json.dumps({"ok": True, "model": MODEL_NAME, "device": ENGINE.device,
                                        "languages": ENGINE.languages, "sample_rate": ENGINE.sr}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != "/generate":
            return self._send(404, b'{"error":"not found"}')
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            text = str(req.get("text") or "")
            if len(text) > MAX_TEXT_CHARS:
                raise ValueError(f"text is longer than {MAX_TEXT_CHARS} characters")
            steps = int(clamp(req.get("num_step"), MIN_STEPS, MAX_STEPS,
                              clamp(os.getenv("DHVAANI_STEPS"), MIN_STEPS, MAX_STEPS, DEFAULT_STEPS)))
            started = time.time()
            audio, rate = EXECUTOR.submit(
                ENGINE.generate, text, req.get("ref_audio_path") or req.get("ref_audio"), req.get("ref_text"),
                req.get("language"), steps, clamp(req.get("speed"), 0.5, 1.5, 1.0),
                int(clamp(req.get("seed"), 0, 2 ** 31 - 1, 666))).result()
            import io

            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, audio, rate, format="WAV", subtype="FLOAT")
            print(f"[dhvaani] {len(text)} chars -> {len(audio) / rate:.1f}s audio in {time.time() - started:.1f}s "
                  f"({steps} steps)", flush=True)
            self._send(200, buf.getvalue(), "audio/wav")
        except (ValueError, json.JSONDecodeError) as e:
            self._send(400, json.dumps({"error": str(e)}).encode())
        except Exception as e:  # noqa: BLE001
            print(f"[dhvaani] generation failed: {type(e).__name__}: {e}", flush=True)
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"[:300]}).encode())


def selftest(model: str, device: str) -> int:
    """Speak one sentence with the model's own Hindi sample as the reference. It proves the model loads and produces
    audible audio on this machine; it does not measure quality (the reference's real transcript is not shipped)."""
    import numpy as np
    started = time.time()
    engine = Engine(model, device)
    load_s = time.time() - started
    ref = os.path.join(engine.model_dir, "samples", "hindi.wav")
    if not os.path.isfile(ref):
        print("[dhvaani] self-test: the model's sample clip is missing; download it again", file=sys.stderr)
        return 1
    started = time.time()
    audio, rate = engine.generate(SELFTEST_TEXT, ref, SELFTEST_REF_TEXT, "hi", 8)
    gen_s = time.time() - started
    seconds = len(audio) / rate
    rms = float(np.sqrt(np.mean(audio ** 2)))
    ok = 1.0 <= seconds <= 30.0 and rms > 0.005
    print(f"[dhvaani] self-test on {engine.device}: loaded in {load_s:.1f}s, {seconds:.1f}s of audio in {gen_s:.1f}s "
          f"(RTF {gen_s / max(seconds, 0.01):.2f}), level {rms:.3f}: {'OK' if ok else 'FAILED'}", flush=True)
    return 0 if ok else 1


def main() -> int:
    global ENGINE, MODEL_NAME, EXECUTOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--download", action="store_true", help="fetch the weights and the vocoder, then exit")
    ap.add_argument("--selftest", action="store_true", help="speak one sentence, then exit")
    args = ap.parse_args()
    MODEL_NAME = args.model

    if args.download:
        return download(args.model)
    if args.selftest:
        return selftest(args.model, args.device)

    _watch_parent()
    EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="dhvaani")
    started = time.time()
    ENGINE = EXECUTOR.submit(Engine, args.model, args.device).result()
    print(f"[dhvaani] loaded {args.model} on {ENGINE.device} in {time.time() - started:.1f}s", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Framework compatibility: packages, PyTorch CUDA, and per-checkpoint architecture / processor / tokenizer / GPU fit.

framework_environment()    versions + CUDA facts (imports only; nothing is loaded on the GPU)
framework_check(...)       FAST check used by eligibility: detected architecture expected for the model, adapter
                           supports it, required packages importable at compatible versions, platform supported,
                           weights loadable by the installed torch/transformers (.bin needs torch >= 2.6 with
                           transformers >= 4.50 -> otherwise a verified safetensors conversion is required),
                           no unreviewed remote code (config.json auto_map) — trust_remote_code is never used
component_check(...)       processor / tokenizer contract: instantiates config, feature extractor / processor and
                           tokenizer from the LOCAL snapshot (local_files_only=True, no weights). Cheap; run by
                           eligibility when deep=True and always by the verification script
gpu_fit(...)               weights + activations vs the GPU's VRAM at the model's precision (RTX 3070: 8 GB)
smoke_forward(...)         loads the weights and decodes one second of silence (verification script only)
"""
from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

MIN_VERSIONS = {"torch": "2.1", "transformers": "4.38", "torchaudio": "2.1", "huggingface_hub": "0.20"}
ARCH_PACKAGES = {"hf_whisper": ("torch", "transformers"), "hf_ctc": ("torch", "transformers", "torchaudio"),
                 "hf_encoder": ("torch", "transformers"), "qwen3_asr": ("torch", "qwen_asr", "vllm"),
                 "nemo": ("torch", "nemo.collections.asr")}
BIN_SAFE_TORCH = "2.6"              # transformers >= 4.50 refuses torch.load on older torch (CVE-2025-32434)
BIN_GUARD_TRANSFORMERS = "4.50"
HALF_UNSAFE_MODEL_TYPES = ("wavlm",)  # gated relative position bias overflows in fp16/bf16 (card + known issue)


def _ver_tuple(v: str) -> tuple:
    out = []
    for part in str(v).split("+")[0].split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out + [0] * (3 - len(out)))


def package_version(name: str) -> tuple[bool, str | None, str]:
    try:
        m = importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        return False, None, f"{name} not importable: {type(e).__name__}: {e}"
    v = getattr(m, "__version__", None)
    if v is None and "." in name:
        try:
            v = importlib.import_module(name.split(".")[0]).__version__
        except Exception:  # noqa: BLE001
            v = None
    base = name.split(".")[0]
    need = MIN_VERSIONS.get(base)
    if need and v and _ver_tuple(v) < _ver_tuple(need):
        return False, v, f"{base} {v} < required {need}"
    return True, v, ""


def bin_load_blocked() -> tuple[bool, str]:
    """(blocked, why): can transformers load a pytorch_model.bin with the installed torch?"""
    okt, vt, _ = package_version("torch")
    okx, vx, _ = package_version("transformers")
    if not (okt and okx and vt and vx):
        return True, "torch / transformers not importable"
    if _ver_tuple(vx) >= _ver_tuple(BIN_GUARD_TRANSFORMERS) and _ver_tuple(vt) < _ver_tuple(BIN_SAFE_TORCH):
        return True, (f"transformers {vx} refuses torch.load of .bin weights with torch {vt} < {BIN_SAFE_TORCH} "
                      "(CVE-2025-32434); use the sha256-verified safetensors conversion (v3.asr.downloads."
                      "convert_verified_bin) instead of upgrading torch")
    return False, f"torch {vt} / transformers {vx} can load .bin"


def framework_environment() -> dict:
    """Versions and CUDA facts. Never allocates model memory."""
    env = {"python": sys.version.split()[0], "platform": sys.platform}
    for n in ("torch", "torchaudio", "transformers", "accelerate", "safetensors", "huggingface_hub", "hf_xet",
              "soundfile", "g2p_en", "qwen_asr", "vllm"):
        ok, v, why = package_version(n)
        env[f"{n}_ok"], env[f"{n}_version"] = ok, v
        if why and ok:
            env[f"{n}_note"] = why
    blocked, why = bin_load_blocked()
    env["bin_weights_loadable"] = not blocked
    env["bin_weights_note"] = why
    try:
        import nemo  # noqa: F401

        env["nemo_ok"], env["nemo_version"] = True, getattr(nemo, "__version__", None)
    except Exception:  # noqa: BLE001
        env["nemo_ok"], env["nemo_version"] = False, None
    try:
        import torch

        env["cuda_available"] = bool(torch.cuda.is_available())
        env["torch_cuda_build"] = getattr(torch.version, "cuda", None)
        if env["cuda_available"]:
            prop = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            env.update({"gpu_name": prop.name, "gpu_total_gb": round(prop.total_memory / 1e9, 2),
                        "gpu_free_gb": round(free / 1e9, 2), "compute_capability": f"{prop.major}.{prop.minor}",
                        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                        "cudnn": torch.backends.cudnn.version()})
    except Exception as e:  # noqa: BLE001
        env["cuda_available"] = False
        env["torch_note"] = f"{type(e).__name__}: {e}"
    return env


@dataclass
class FrameworkResult:
    ok: bool
    architecture: str
    detail: str = ""
    warnings: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    remedies: list[str] = field(default_factory=list)


def _weights_present(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    out = set()
    for n in ("model.safetensors", "pytorch_model.bin"):
        q = p / n
        try:
            if q.exists() and q.is_file():
                out.add(n)
        except OSError:
            pass
    if list(p.glob("model-*.safetensors")):
        out.add("model.safetensors")
    return out


def framework_check(spec, architecture: str, path: str | None = None, cfg=None) -> FrameworkResult:
    from .model_registry import ADAPTER_ARCHITECTURES
    from .snapshots import read_config, remote_code_files

    ch, warn, bad, rem = {}, [], [], []
    exp = list(spec.expected_architectures or [])
    ch["architecture_expected"] = (architecture in exp) if exp else None
    if exp and architecture not in exp:
        bad.append(f"snapshot architecture '{architecture}' not in expected {exp} (wrong checkpoint?)")
    elif not exp:
        warn.append("registry lists no expected_architectures")
    c = read_config(path) if path else {}
    if path and spec.expected_model_type and c.get("model_type") and c["model_type"] != spec.expected_model_type:
        bad.append(f"config model_type {c['model_type']} != expected {spec.expected_model_type}")
        ch["architecture_expected"] = False
    if path and spec.expected_class and c.get("architectures") and spec.expected_class not in c["architectures"]:
        bad.append(f"config architectures {c['architectures']} do not include {spec.expected_class}")
        ch["architecture_expected"] = False
    sup = ADAPTER_ARCHITECTURES.get(spec.adapter, ())
    ch["adapter_compatible"] = architecture in sup
    if architecture not in sup:
        bad.append(f"adapter '{spec.adapter}' cannot drive architecture '{architecture}' (supports {list(sup)})")
    # ---- remote code: never trusted unless the exact files were reviewed (sha256 recorded in the registry)
    rc = remote_code_files(path) if path else {}
    if rc:
        reviewed = dict(getattr(spec, "reviewed_remote_code", {}) or {})
        files = sorted({str(v).split(".")[0] + ".py" for v in rc.values() if isinstance(v, str)})
        ok_rc = bool(reviewed) and all(f in reviewed for f in files)
        if not ok_rc:
            bad.append(f"config.json auto_map requires remote code {files}; trust_remote_code is never enabled "
                       "for unreviewed code")
            rem.append("review the exact remote code files and record their sha256 under reviewed_remote_code")
        ch["remote_code"] = ok_rc
    else:
        ch["remote_code"] = True
    # ---- packages / platform
    pk = ARCH_PACKAGES.get(architecture, ())
    pk_ok = True
    for n in pk:
        ok, v, why = package_version(n)
        if not ok:
            pk_ok = False
            bad.append(why)
    rr = dict(getattr(spec, "runtime_requirements", {}) or {})
    plats = rr.get("supported_platforms")
    allow_tf = bool(cfg.get_path("v3.asr.qwen3_asr.allow_transformers_backend", False)) if cfg is not None else False
    if architecture == "qwen3_asr":
        if plats and not any(sys.platform.startswith(pl) for pl in plats) and not allow_tf:
            bad.append(f"runtime backend {spec.runtime_backend} supports {plats}; this is {sys.platform} (vLLM has no "
                       "native Windows build)")
            rem.append("run P24 for this model inside WSL2 / Linux: scripts/v3_pasketti_word_wsl.sh "
                       "(same repo, same cache/v3/asr output)")
            pk_ok = False
        if allow_tf:
            warn.append("qwen3_asr transformers backend enabled although the card says it is not supported "
                        "(experimental; outputs are tagged backend=qwen_asr_transformers)")
    ch["packages_importable"] = pk_ok if pk else False
    if pk and not pk_ok and not rem:
        rem.append("run scripts\\v3_install_asr_env.ps1")
    # ---- weights loadable with the installed torch
    have = _weights_present(path) if path else set()
    if path and architecture in ("hf_whisper", "hf_ctc", "hf_encoder"):
        if "model.safetensors" not in have and "pytorch_model.bin" in have:
            blocked, why = bin_load_blocked()
            if blocked:
                bad.append(why)
                rem.append(f"python -m wildspeech v3-install --model {spec.key} --execute  (verifies the .bin sha256 "
                           "against the registry, converts to safetensors locally)")
            ch["weights_loadable"] = not blocked
        else:
            ch["weights_loadable"] = bool(have)
    else:
        ch["weights_loadable"] = None
    if architecture == "hf_ctc" and pk_ok:
        try:
            import torchaudio.functional as AF

            if not hasattr(AF, "forced_align"):
                warn.append("torchaudio has no functional.forced_align: expected-text alignment columns will be empty")
        except Exception:  # noqa: BLE001
            pass
    if (c.get("model_type") in HALF_UNSAFE_MODEL_TYPES) and str(spec.precision or "auto").lower() not in ("fp32",):
        warn.append(f"{c.get('model_type')} in half precision can produce NaN; the adapter forces fp32")
    return FrameworkResult(not bad, architecture, "; ".join(bad), warn, ch, rem)


# ======================================================================================= GPU readiness
def precision_bytes(spec, env: dict) -> int:
    p = str(getattr(spec, "precision", None) or "auto").lower()
    if p == "fp32":
        return 4
    return 2 if env.get("cuda_available") else 4


def gpu_fit(spec, env: dict, cfg=None) -> dict:
    """Static VRAM estimate: weights at the model's precision + CUDA context + one clip of activations."""
    b = (cfg.get_path("v3.asr.batch", {}) if cfg is not None else {}) or {}
    params = float(spec.expected_params_millions or 0) * 1e6
    w_gb = params * precision_bytes(spec, env) / 1e9
    large = params >= float(b.get("large_model_params", 300_000_000))
    act = float(b.get("activation_gb_per_clip_large", 0.9) if large else b.get("activation_gb_per_clip_small", 0.35))
    need = w_gb + float(b.get("cuda_context_gb", 0.8)) + act
    if spec.runtime_backend == "qwen_asr_vllm":
        need = w_gb + float(b.get("cuda_context_gb", 0.8)) + 1.5      # vLLM KV cache for short utterances
    total = env.get("gpu_total_gb")
    usable = (float(total) * float(b.get("vram_headroom", 0.80))) if total else None
    return {"weights_gb": round(w_gb, 2), "estimated_need_gb": round(need, 2), "gpu_total_gb": total,
            "usable_gb": round(usable, 2) if usable else None,
            "fits": (need <= usable) if usable else None, "cuda": bool(env.get("cuda_available")),
            "cpu_feasible": params < 400e6 and spec.runtime_backend != "qwen_asr_vllm"}


# ======================================================================================= deep checks
def _tokenizer_class(path: str) -> str | None:
    p = Path(path) / "tokenizer_config.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("tokenizer_class")
    except Exception:  # noqa: BLE001
        return None


def load_ctc_tokenizer(path: str, output: str, kw: dict):
    """CTC tokenizer resolved from the snapshot's own tokenizer_config (never guessed):
    Wav2Vec2PhonemeCTCTokenizer (do_phonemize=False: the model's phone symbols are used verbatim) or the plain
    Wav2Vec2CTCTokenizer / AutoTokenizer (Pasketti IPA characters, LibriSpeech characters)."""
    cls = _tokenizer_class(path) or ""
    if "Phoneme" in cls:
        from transformers import Wav2Vec2PhonemeCTCTokenizer

        return Wav2Vec2PhonemeCTCTokenizer.from_pretrained(path, do_phonemize=False, **kw)
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path, **kw)
    except Exception:  # noqa: BLE001 - snapshots without tokenizer_config
        from transformers import Wav2Vec2CTCTokenizer

        return Wav2Vec2CTCTokenizer.from_pretrained(path, **kw)


def component_check(spec, path: str, architecture: str) -> dict:
    """Load config / processor / tokenizer from the LOCAL snapshot (no weights) and check the adapter contract."""
    out = {"model_key": spec.key, "architecture": architecture, "path": path, "ok": False, "problems": [], "notes": []}
    kw = {"local_files_only": True}
    P, N = out["problems"], out["notes"]
    sr_need = int(getattr(spec, "sample_rate", 16000) or 16000)
    try:
        if architecture == "hf_whisper":
            from transformers import AutoConfig, WhisperProcessor

            cfg = AutoConfig.from_pretrained(path, **kw)
            proc = WhisperProcessor.from_pretrained(path, **kw)
            fe, tok = proc.feature_extractor, proc.tokenizer
            out.update({"model_type": cfg.model_type, "sampling_rate": fe.sampling_rate, "n_mels": getattr(fe, "feature_size", None),
                        "vocab_size": len(tok), "tokenizer_class": type(tok).__name__,
                        "processor_class": type(proc).__name__})
            if fe.sampling_rate != sr_need:
                P.append(f"processor sampling_rate {fe.sampling_rate} != {sr_need}")
            multi = "<|en|>" in tok.get_vocab() and len(tok) > 51864
            out["multilingual_tokenizer"] = multi
            if spec.key.endswith("_en") and multi:
                P.append("expected an English-only (.en) tokenizer but found a multilingual vocabulary")
            if spec.output == "phones":
                N.append("seq2seq phone recogniser: decoded strings are parsed to ARPAbet (parse_phone_string)")
        elif architecture == "hf_ctc":
            from transformers import AutoConfig, AutoFeatureExtractor

            cfg = AutoConfig.from_pretrained(path, **kw)
            fe = AutoFeatureExtractor.from_pretrained(path, **kw)
            tok = load_ctc_tokenizer(path, spec.output, kw)
            vocab = tok.get_vocab()
            out.update({"model_type": cfg.model_type, "architectures": ",".join(cfg.architectures or []),
                        "sampling_rate": getattr(fe, "sampling_rate", None), "vocab_size": len(vocab),
                        "tokenizer_class": type(tok).__name__, "feat_extract_norm": getattr(cfg, "feat_extract_norm", None),
                        "return_attention_mask": getattr(fe, "return_attention_mask", None),
                        "pad_token": tok.pad_token, "word_delimiter": "|" if "|" in vocab else (" " if " " in vocab else None)})
            if getattr(fe, "sampling_rate", sr_need) != sr_need:
                P.append(f"feature extractor sampling_rate {fe.sampling_rate} != {sr_need}")
            if tok.pad_token_id is None:
                P.append("CTC tokenizer has no pad/blank token")
            if not any("ForCTC" in a for a in (cfg.architectures or [])):
                P.append(f"config architectures {cfg.architectures} contain no *ForCTC head")
            if spec.output == "phones":
                from ..phonetic.g2p import ARPA_CONSONANTS, ARPA_VOWELS
                from .ctc_adapter import ipa_sequence_to_arpa

                arpa = ARPA_VOWELS | ARPA_CONSONANTS
                syms = [k for k in vocab if not (k.startswith("<") and k.endswith(">")) and k not in ("|", " ")]
                covered = {a for s in syms for a in ipa_sequence_to_arpa([s], merge=False)}
                if str(getattr(spec, "phone_inventory", "")) == "ipa_characters":     # CH = t+ʃ, AY = a+ɪ ...
                    from ..phonetic.g2p import ARPA_TO_IPA

                    covered |= {a for a, cs in ARPA_TO_IPA.items() if any(all(ch in vocab for ch in c if ch != "ː") for c in cs)}
                out.update({"phone_symbols": len(syms), "arpabet_inventory_covered": len(covered & arpa),
                            "arpabet_inventory_size": len(arpa)})
                if len(covered & arpa) < 0.6 * len(arpa):
                    P.append(f"phone vocabulary covers only {len(covered & arpa)}/{len(arpa)} ARPAbet phones")
            elif spec.output == "chars":
                letters = [k for k in vocab if len(k) == 1 and k.isalpha()]
                out["letters"] = len(letters)
                if len(letters) < 26:
                    P.append(f"character vocabulary has {len(letters)} letters (< 26)")
        elif architecture == "hf_encoder":
            from transformers import AutoConfig, AutoFeatureExtractor

            cfg = AutoConfig.from_pretrained(path, **kw)
            fe = AutoFeatureExtractor.from_pretrained(path, **kw)
            out.update({"model_type": cfg.model_type, "architectures": ",".join(cfg.architectures or []),
                        "hidden_size": getattr(cfg, "hidden_size", None),
                        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
                        "sampling_rate": getattr(fe, "sampling_rate", None),
                        "feat_extract_norm": getattr(cfg, "feat_extract_norm", None)})
            if getattr(fe, "sampling_rate", sr_need) != sr_need:
                P.append(f"feature extractor sampling_rate {fe.sampling_rate} != {sr_need}")
            nl = int(getattr(cfg, "num_hidden_layers", 0) or 0)
            bad_layers = [x for x in (spec.encoder_layers or []) if x < 0 or x > nl]
            if bad_layers:
                P.append(f"encoder_layers {bad_layers} outside 0..{nl}")
        elif architecture == "qwen3_asr":
            for f in ("config.json", "preprocessor_config.json"):
                if not (Path(path) / f).exists():
                    P.append(f"missing {f}")
            c = json.loads((Path(path) / "config.json").read_text(encoding="utf-8")) if (Path(path) / "config.json").exists() else {}
            pp = Path(path) / "preprocessor_config.json"
            pc = json.loads(pp.read_text(encoding="utf-8")) if pp.exists() else {}
            out.update({"model_type": c.get("model_type"), "architectures": ",".join(c.get("architectures") or []),
                        "sampling_rate": pc.get("sampling_rate")})
            if pc.get("sampling_rate") not in (None, sr_need):
                P.append(f"processor sampling_rate {pc.get('sampling_rate')} != {sr_need}")
            N.append("Qwen3-ASR components are instantiated by qwen_asr at load time (smoke test verifies them)")
        elif architecture == "nemo":
            N.append("NeMo checkpoint: components are restored with the model (use --load-weights to verify)")
        else:
            P.append(f"no component rules for architecture {architecture}")
    except Exception as e:  # noqa: BLE001
        P.append(f"{type(e).__name__}: {e}")
    out["ok"] = not P
    out["problems"] = "; ".join(P)
    out["notes"] = "; ".join(N)
    return out


def smoke_forward(spec, rec, cfg, device: str) -> dict:
    """Load weights from the local snapshot and decode 1 s of low-level noise. Only for the verification script."""
    import time

    import numpy as np

    from .base import ModelSlot
    from .inference import build_adapter

    t0 = time.perf_counter()
    ad = build_adapter(spec, rec, cfg, device, technical_only=True)     # synthetic audio only
    wave = (np.random.default_rng(0).standard_normal(16000) * 1e-3).astype(np.float32)
    with ModelSlot(spec.key):
        ad.load()
        info = ad.version_info()
        res = ad.transcribe_batch([wave], [""], 16000)
        ad.unload()
    r = res[0] if res else None
    return {"model_key": spec.key, "ok": len(res) == 1 and not (r.error if r else "x"), "seconds": round(time.perf_counter() - t0, 2),
            "dtype": info.get("dtype"), "n_params": info.get("n_params"), "device": device,
            "token_probs": info.get("token_probs"), "frame_posteriors": info.get("frame_posteriors"),
            "embedding_dim": int(len(r.embedding)) if (r is not None and r.embedding is not None) else 0}

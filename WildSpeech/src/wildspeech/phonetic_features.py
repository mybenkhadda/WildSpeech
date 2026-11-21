"""Section 11 - linguistic item complexity and phonetic intelligence.

Optional local ASR runs only when ``enable_asr`` and ``enable_phonetic_analysis``
are on AND each model entry in ``CONFIG['asr_models']`` is explicitly
configured and marked license-verified and competition-eligible by the user.
Audio is never sent to remote services. ASR disagreement with the expected
text is an uncertain proxy for pronunciation differences, not a verified
child error.

ASR_MODEL_SPEC (one dict per model)::

    {"alias": "whisper_small", "type": "whisper" | "ctc" | "ctc_phoneme",
     "checkpoint": "<local path or hub id>", "license_verified": True,
     "eligible_for_competition": True, "language": "en", "batch_size": 16}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from . import visualization as viz
from .linguistic_features import ARPABET_VOWELS, G2P, item_linguistic_table, tokenize_words

ASR_MODEL_SPEC = {"alias": str, "type": str, "checkpoint": str, "license_verified": bool,
                  "eligible_for_competition": bool}


# ----------------------------------------------------------------- alignment
def align(ref: list, hyp: list) -> list:
    """Levenshtein alignment; returns ops (op, ref_token, hyp_token) with op in =,S,D,I."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=int)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + cost)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def lcs_length(a: list, b: list) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[-1]))
        prev = cur
    return prev[-1]


def error_summary(ref: list, hyp: list, prefix: str) -> dict:
    ops = align(ref, hyp)
    counts = {k: sum(o[0] == k for o in ops) for k in ("=", "S", "D", "I")}
    n = max(len(ref), 1)
    out = {f"{prefix}_er": (counts["S"] + counts["D"] + counts["I"]) / n if ref else np.nan,
           f"{prefix}_sub_rate": counts["S"] / n, f"{prefix}_del_rate": counts["D"] / n, f"{prefix}_ins_rate": counts["I"] / n,
           f"{prefix}_completeness": 1 - counts["D"] / n if ref else np.nan,
           f"{prefix}_order_preservation": lcs_length(ref, hyp) / n if ref else np.nan}
    first_err = next((k for k, o in enumerate(ops) if o[0] != "="), None)
    out[f"{prefix}_first_error_pos"] = first_err / max(len(ops), 1) if first_err is not None else np.nan
    return out, ops


def self_correction_count(words: list) -> int:
    """Repeated words or partial-word restarts ('c cat', 'the the')."""
    count = 0
    for a, b in zip(words[:-1], words[1:]):
        if a == b or (len(a) < len(b) and b.startswith(a)):
            count += 1
    return count


# -------------------------------------------------------------------- adapters
class ASRAdapter:
    def __init__(self, spec: dict, device: str):
        self.spec = spec
        self.device = device
        self.alias = spec.get("alias") or spec["type"]

    def transcribe(self, waves: list, sr: int) -> list:
        raise NotImplementedError


class WhisperAdapter(ASRAdapter):
    def __init__(self, spec: dict, device: str, local_only: bool):
        super().__init__(spec, device)
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.processor = WhisperProcessor.from_pretrained(spec["checkpoint"], local_files_only=local_only)
        self.model = WhisperForConditionalGeneration.from_pretrained(spec["checkpoint"], local_files_only=local_only).to(device).eval()

    def transcribe(self, waves: list, sr: int) -> list:
        torch = self.torch
        inputs = self.processor(waves, sampling_rate=sr, return_tensors="pt")
        feats = inputs.input_features.to(self.device)
        with torch.no_grad():
            gen = self.model.generate(feats, output_scores=True, return_dict_in_generate=True,
                                      language=self.spec.get("language", "en"), task="transcribe")
        texts = self.processor.batch_decode(gen.sequences, skip_special_tokens=True)
        scores = torch.stack(gen.scores, dim=1).float().softmax(-1)  # batch x steps x vocab
        tok = gen.sequences[:, -scores.shape[1]:]
        conf = scores.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
        ent = -(scores * (scores + 1e-12).log()).sum(-1)
        special = torch.tensor(self.processor.tokenizer.all_special_ids, device=tok.device)
        out = []
        for k in range(len(texts)):
            mask = ~torch.isin(tok[k], special)
            c, e = conf[k][mask].cpu().numpy(), ent[k][mask].cpu().numpy()
            out.append({"text": texts[k].strip(), "token_conf_mean": float(c.mean()) if len(c) else np.nan,
                        "token_conf_min": float(c.min()) if len(c) else np.nan,
                        "token_entropy_mean": float(e.mean()) if len(e) else np.nan,
                        "seq_confidence": float(np.exp(np.log(c + 1e-12).mean())) if len(c) else np.nan})
        return out


class CTCAdapter(ASRAdapter):
    def __init__(self, spec: dict, device: str, local_only: bool):
        super().__init__(spec, device)
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(spec["checkpoint"], local_files_only=local_only)
        self.model = AutoModelForCTC.from_pretrained(spec["checkpoint"], local_files_only=local_only).to(device).eval()
        self.blank = getattr(self.processor.tokenizer, "pad_token_id", 0) or 0

    def transcribe(self, waves: list, sr: int, expected: list | None = None) -> list:
        torch = self.torch
        inputs = self.processor(waves, sampling_rate=sr, return_tensors="pt", padding=True)
        values = inputs.input_values.to(self.device)
        attn = inputs.get("attention_mask")
        with torch.no_grad():
            logits = self.model(values, attention_mask=attn.to(self.device) if attn is not None else None).logits.float()
        logp = logits.log_softmax(-1)
        ids = logp.argmax(-1)
        texts = self.processor.batch_decode(ids)
        frame_s = values.shape[1] / sr / logits.shape[1]
        out = []
        for k in range(len(texts)):
            n_frames = logits.shape[1]
            if attn is not None:
                n_frames = int(np.ceil(attn[k].sum().item() / values.shape[1] * logits.shape[1]))
            lp = logp[k, :n_frames]
            p = lp.exp()
            top = p.max(-1).values
            nonblank = ids[k, :n_frames] != self.blank
            ent = -(p * lp).sum(-1)
            rec = {"text": texts[k].strip(),
                   "align_conf_mean": float(top[nonblank].mean()) if nonblank.any() else np.nan,
                   "align_entropy_mean": float(ent.mean()),
                   "nonblank_frame_frac": float(nonblank.float().mean()),
                   "phone_duration_mean_s": float(nonblank.sum().item() * frame_s / max(len(texts[k].split()), 1))}
            rec["seq_confidence"] = rec["align_conf_mean"]
            if expected is not None and expected[k]:
                target = self.processor.tokenizer(expected[k], return_tensors="pt").input_ids[0]
                target = target[target != self.blank]
                if len(target) and len(target) <= n_frames:
                    loss = torch.nn.functional.ctc_loss(lp.unsqueeze(1).cpu(), target.unsqueeze(0), [n_frames], [len(target)],
                                                        blank=self.blank, reduction="sum", zero_infinity=True)
                    rec["expected_loglik_per_token"] = float(-loss.item() / len(target))
            out.append(rec)
        return out


def build_adapter(spec: dict, cfg: dict):
    if not spec.get("license_verified") or not spec.get("eligible_for_competition"):
        raise PermissionError(f"ASR model {spec.get('alias')} not marked license_verified and eligible_for_competition")
    device = "cpu"
    try:
        import torch

        if cfg.get("use_gpu") and torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass
    local = bool(cfg.get("hf_local_files_only", True))
    if spec["type"] == "whisper":
        return WhisperAdapter(spec, device, local)
    if spec["type"] in {"ctc", "ctc_phoneme"}:
        return CTCAdapter(spec, device, local)
    raise ValueError(f"unknown ASR type {spec['type']}")


def run_asr(ctx, adapter, clips: pd.DataFrame, split: str) -> pd.DataFrame:
    """Batched local ASR with per-chunk caching and CUDA OOM recovery."""
    from .audio_processing import decode_audio
    from . import audio_manifest as am
    from .io_utils import atomic_write_parquet
    from .logging_utils import progress, release_memory

    cfg = ctx.config
    cache = ctx.paths.cache_dir("phonetic") / f"{split}_{adapter.alias}"
    cache.mkdir(parents=True, exist_ok=True)
    done = set()
    for p in cache.glob("chunk_*.parquet"):
        done |= set(pd.read_parquet(p, columns=["clip_id"])["clip_id"])
    todo = clips[~clips["clip_id"].isin(done) & clips["audio_member"].notna()]
    archive = ctx.data_root() / ("train_audio.tar" if split == "train" else "test_audio.tar")
    extracted = ctx.paths.extracted_train if split == "train" else ctx.paths.extracted_test
    batch = int(adapter.spec.get("batch_size", cfg["audio_batch_size"]))
    chunk_size = max(batch * 8, 64)
    sr = 16000
    start_idx = len(list(cache.glob("chunk_*.parquet")))
    for ci, start in enumerate(progress(range(0, len(todo), chunk_size), desc=f"ASR {adapter.alias}")):
        chunk = todo.iloc[start:start + chunk_size]
        members = chunk["audio_member"].tolist()
        use_files = all((extracted / m).exists() for m in members)
        blobs = {} if use_files else am.read_member_bytes(archive, members)
        waves, ids, expected = [], [], []
        for _, r in chunk.iterrows():
            src = str(extracted / r["audio_member"]) if use_files else blobs.get(r["audio_member"])
            if src is None:
                continue
            try:
                y, _ = decode_audio(src, sr)
            except Exception:
                continue
            waves.append(y[: int(float(cfg["embedding_max_seconds"]) * sr)])
            ids.append(r["clip_id"])
            expected.append(str(r.get("expected_text", "") or ""))
        rows, k, bs = [], 0, batch
        while k < len(waves):
            try:
                part = waves[k:k + bs]
                if isinstance(adapter, CTCAdapter):
                    res = adapter.transcribe(part, sr, expected[k:k + bs])
                else:
                    res = adapter.transcribe(part, sr)
                for cid, rec in zip(ids[k:k + bs], res):
                    rows.append({"clip_id": cid, **rec})
                k += bs
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and bs > 1:
                    release_memory()
                    bs = max(1, bs // 2)
                    ctx.logger.warning("CUDA OOM in ASR; reducing batch to %d", bs)
                    continue
                raise
        atomic_write_parquet(pd.DataFrame(rows), cache / f"chunk_{start_idx + ci:05d}.parquet")
    parts = [pd.read_parquet(p) for p in sorted(cache.glob("chunk_*.parquet"))]
    return pd.concat(parts, ignore_index=True).drop_duplicates("clip_id", keep="last") if parts else pd.DataFrame(columns=["clip_id"])


def compare_expected(asr: pd.DataFrame, clips: pd.DataFrame, g2p: G2P, alias: str, phoneme_output: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = asr.merge(clips[["clip_id", "expected_text", "task_family"]], on="clip_id", how="left")
    rows, confusion = [], []
    p = f"ph_{alias}"
    for _, r in d.iterrows():
        exp_words = tokenize_words(r.get("expected_text", ""))
        hyp_text = str(r.get("text", "") or "")
        hyp_words = tokenize_words(hyp_text)
        rec = {"clip_id": r["clip_id"], f"{p}_text": hyp_text}
        w, _ = error_summary(exp_words, hyp_words, f"{p}_word")
        c, _ = error_summary(list("".join(exp_words)), list("".join(hyp_words)), f"{p}_char")
        exp_ph = [x for wd in exp_words for x in g2p.word(wd)]
        hyp_ph = hyp_text.upper().split() if phoneme_output else [x for wd in hyp_words for x in g2p.word(wd)]
        ph, ops = error_summary(exp_ph, hyp_ph, f"{p}_phone")
        rec.update(w)
        rec.update(c)
        rec.update(ph)
        cons = [o for o in ops if o[1] is not None and o[1] not in ARPABET_VOWELS]
        vow = [o for o in ops if o[1] is not None and o[1] in ARPABET_VOWELS]
        rec[f"{p}_consonant_accuracy"] = float(np.mean([o[0] == "=" for o in cons])) if cons else np.nan
        rec[f"{p}_vowel_accuracy"] = float(np.mean([o[0] == "=" for o in vow])) if vow else np.nan
        rec[f"{p}_self_corrections"] = float(self_correction_count(hyp_words))
        rec[f"{p}_hyp_len_ratio"] = len(hyp_words) / len(exp_words) if exp_words else np.nan
        for key in ("token_conf_mean", "token_conf_min", "token_entropy_mean", "seq_confidence", "align_conf_mean",
                    "align_entropy_mean", "nonblank_frame_frac", "phone_duration_mean_s", "expected_loglik_per_token"):
            if key in r and pd.notna(r[key]):
                rec[f"{p}_{key}"] = float(r[key])
        rows.append(rec)
        for o in ops:
            if o[0] != "=":
                confusion.append({"task_family": r.get("task_family"), "op": o[0], "expected": o[1], "observed": o[2]})
    return pd.DataFrame(rows), pd.DataFrame(confusion)


def stage_phonetic_intelligence(ctx) -> dict:
    cfg = ctx.config
    outputs: list = []
    items = ctx.load_df("item_master")
    summary: dict = {}
    if cfg.get("enable_linguistic_features") or cfg.get("enable_phonetic_analysis"):
        table, info = item_linguistic_table(items, cfg)
        ctx.save_df(table, "item_linguistic_features", also_csv=True)
        ctx.save_json(info, "linguistic_status")
        outputs += ["item_linguistic_features", "linguistic_status"]
        summary["linguistic"] = info
        if len(table) and "l_n_phonemes" in table.columns:
            fig, axes = viz.new_figure(1, 3, (4.5, 3.6))
            for ax, c in zip(axes.ravel(), ("l_n_phonemes", "l_n_syllables", "l_max_cluster_len")):
                viz.hist_kde(ax, table[c], c, bins=20)
            ctx.figure(fig, "item_linguistic_complexity", "Expected-text complexity of items",
                       "How complex are the items linguistically?", "item_linguistic_features", int(len(table)),
                       f"G2P backend: {info['g2p_backend']}{' (approximate rules)' if info.get('approximate') else ''}.")
    specs = cfg.get("asr_models") or []
    if not (cfg.get("enable_asr") and cfg.get("enable_phonetic_analysis")):
        ctx.save_json({"status": "SKIPPED_DISABLED", "reason": "enable_asr and enable_phonetic_analysis must both be True"}, "asr_status")
        return {"outputs": outputs + ["asr_status"], "summary": summary}
    if not specs:
        ctx.save_json({"status": "NOT_CONFIGURED", "reason": "no entries in CONFIG['asr_models']"}, "asr_status")
        return {"outputs": outputs + ["asr_status"], "summary": summary}
    g2p = G2P(cfg.get("g2p_backend", "auto"))
    status = []
    for split in ["train"] + (["test"] if cfg.get("allow_test_inference") else []):
        clips = da.clips(ctx, split)
        merged = None
        confusions = []
        for spec in specs:
            try:
                adapter = build_adapter(spec, cfg)
            except Exception as exc:
                status.append({"alias": spec.get("alias"), "split": split, "status": "UNAVAILABLE", "reason": str(exc)})
                continue
            asr = run_asr(ctx, adapter, clips, split)
            feats, conf = compare_expected(asr, clips, g2p, adapter.alias, spec["type"] == "ctc_phoneme")
            merged = feats if merged is None else merged.merge(feats, on="clip_id", how="outer")
            confusions.append(conf.assign(alias=adapter.alias))
            status.append({"alias": adapter.alias, "split": split, "status": "OK", "n": int(len(feats))})
            del adapter
        if merged is not None:
            name = "clip_phonetic_features" if split == "train" else "clip_phonetic_features_test"
            ctx.save_df(merged, name)
            outputs.append(name)
        if split == "train" and confusions:
            conf = pd.concat(confusions, ignore_index=True)
            ctx.save_df(conf, "phoneme_confusions")
            outputs.append("phoneme_confusions")
            _phonetic_figures(ctx, conf, merged)
    ctx.save_json({"status": "COMPLETED", "models": status,
                   "caveat": "ASR disagreement is an uncertain proxy, not a verified child error."}, "asr_status")
    outputs.append("asr_status")
    summary["asr"] = status
    return {"outputs": outputs, "summary": summary}


def _phonetic_figures(ctx, conf: pd.DataFrame, feats: pd.DataFrame) -> None:
    subs = conf[conf["op"] == "S"]
    if len(subs):
        top = subs["expected"].value_counts().head(30).index
        M = pd.crosstab(subs["expected"], subs["observed"]).reindex(index=top).fillna(0)
        M = M.loc[:, M.sum().sort_values(ascending=False).index[:30]]
        fig, axes = viz.new_figure(1, 1, (9, 8))
        viz.heatmap(axes[0, 0], M, "Phoneme substitutions (expected rows, recognised columns)", cmap="viridis", cbar_label="count")
        ctx.figure(fig, "phoneme_confusion", "Phoneme confusion matrix", "Which expected sounds are recognised differently?",
                   "phoneme_confusions", int(len(subs)), "ASR-based proxy; not verified child errors.")
    fig, axes = viz.new_figure(1, 1)
    conf.groupby(["task_family", "op"]).size().unstack(fill_value=0).plot(kind="bar", stacked=True, ax=axes[0, 0])
    viz.finalize_axis(axes[0, 0], "Error-type distribution by task", "task", "operations", int(len(conf)))
    ctx.figure(fig, "phonetic_error_types", "Task-specific phonetic error profiles", "Which error types dominate per task?",
               "phoneme_confusions", int(len(conf)), "S=substitution, D=deletion, I=insertion (ASR proxy).")
    pos = [c for c in feats.columns if c.endswith("_phone_first_error_pos")]
    conf_cols = [c for c in feats.columns if c.endswith(("_seq_confidence", "_align_conf_mean"))]
    if pos or conf_cols:
        fig, axes = viz.new_figure(1, 2)
        if pos:
            viz.hist_kde(axes[0, 0], feats[pos[0]], "normalised first-error position")
        if conf_cols:
            viz.hist_kde(axes[0, 1], feats[conf_cols[0]], conf_cols[0])
        ctx.figure(fig, "phonetic_error_position_confidence", "Error positions and recognition confidence",
                   "Where do deviations occur and how confident is recognition?", "clip_phonetic_features",
                   int(len(feats)), "Confidence values are model-specific and uncalibrated.")

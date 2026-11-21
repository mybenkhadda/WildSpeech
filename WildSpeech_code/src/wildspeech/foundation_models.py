"""Optional Section 23 - speech foundation-model exploration (disabled by default).

Frozen-encoder embeddings (Whisper encoder, wav2vec2, HuBERT, WavLM or a
suitable child-speech model) with layer selection, mean / std / quantile /
speech-active-frame pooling, task-conditioned and student-level hierarchical
pooling, and train-fitted PCA. No fine-tuning in P01.

EMBEDDING_MODEL_SPEC (one dict per model)::

    {"alias": "wavlm_base", "type": "whisper_encoder" | "ssl",
     "checkpoint": "<local path or hub id>", "layers": [6, 12],
     "license_verified": True, "eligible_for_competition": True,
     "data_overlap_checked": True, "batch_size": 16}

Each entry must be explicitly license-verified, competition-eligible and
checked for pretraining-data overlap by the user before it is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from . import visualization as viz

EMBEDDING_MODEL_SPEC = {"alias": str, "type": str, "checkpoint": str, "layers": list, "license_verified": bool,
                        "eligible_for_competition": bool, "data_overlap_checked": bool}


class EmbeddingExtractor:
    def __init__(self, spec: dict, cfg: dict):
        for flag in ("license_verified", "eligible_for_competition", "data_overlap_checked"):
            if not spec.get(flag):
                raise PermissionError(f"embedding model {spec.get('alias')} requires {flag}=True")
        import torch

        self.torch = torch
        self.spec = spec
        self.alias = spec.get("alias") or spec["type"]
        self.device = "cuda" if cfg.get("use_gpu") and torch.cuda.is_available() else "cpu"
        local = bool(cfg.get("hf_local_files_only", True))
        if spec["type"] == "whisper_encoder":
            from transformers import WhisperFeatureExtractor, WhisperModel

            self.fe = WhisperFeatureExtractor.from_pretrained(spec["checkpoint"], local_files_only=local)
            self.model = WhisperModel.from_pretrained(spec["checkpoint"], local_files_only=local).encoder.to(self.device).eval()
            self.frame_s = 0.02
        else:
            from transformers import AutoFeatureExtractor, AutoModel

            self.fe = AutoFeatureExtractor.from_pretrained(spec["checkpoint"], local_files_only=local)
            self.model = AutoModel.from_pretrained(spec["checkpoint"], local_files_only=local).to(self.device).eval()
            self.frame_s = 0.02
        self.layers = list(spec.get("layers") or [-1])
        self.amp = self.device == "cuda"

    def embed(self, waves: list, sr: int, masks: list) -> list:
        torch = self.torch
        if self.spec["type"] == "whisper_encoder":
            inputs = self.fe(waves, sampling_rate=sr, return_tensors="pt")
            x = inputs.input_features.to(self.device)
            kwargs = {}
        else:
            inputs = self.fe(waves, sampling_rate=sr, return_tensors="pt", padding=True, return_attention_mask=True)
            x = inputs.input_values.to(self.device)
            kwargs = {"attention_mask": inputs["attention_mask"].to(self.device)} if "attention_mask" in inputs else {}
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=self.amp):
            out = self.model(x, output_hidden_states=True, **kwargs)
        hidden = out.hidden_states
        results = []
        for k, w in enumerate(waves):
            n_frames = int(min(hidden[0].shape[1], np.ceil(len(w) / sr / self.frame_s)))
            rec = {}
            vad = masks[k]
            for layer in self.layers:
                H = hidden[layer][k, :n_frames].float().cpu().numpy()
                tag = f"L{layer if layer >= 0 else len(hidden) + layer}"
                rec[tag] = {"mean": H.mean(0), "std": H.std(0), "q10": np.quantile(H, 0.1, axis=0), "q90": np.quantile(H, 0.9, axis=0)}
                if vad is not None:
                    idx = np.clip((np.arange(n_frames) * self.frame_s / vad["hop_s"]).astype(int), 0, len(vad["mask"]) - 1)
                    sel = vad["mask"][idx]
                    rec[tag]["speech_mean"] = H[sel].mean(0) if sel.sum() >= 2 else np.full(H.shape[1], np.nan)
            results.append(rec)
        return results


def _vad_masks(ctx, clip_ids: list, split: str) -> dict:
    name = "vad_segments" if split == "train" else "vad_segments_test"
    segs = ctx.load_df(name, required=False)
    if segs is None:
        return {}
    hop_s = ctx.config["hop_length"] / ctx.config["sample_rate"]
    out = {}
    for cid, g in segs[segs["clip_id"].isin(set(clip_ids))].groupby("clip_id"):
        n = int(np.ceil(g["end_s"].max() / hop_s)) + 1
        m = np.zeros(n, bool)
        for s, e in zip(g["start_s"], g["end_s"]):
            m[int(s / hop_s):int(e / hop_s) + 1] = True
        out[cid] = {"mask": m, "hop_s": hop_s}
    return out


def extract_embeddings(ctx, extractor: EmbeddingExtractor, split: str) -> pd.DataFrame:
    from . import audio_manifest as am
    from .audio_processing import decode_audio
    from .io_utils import atomic_write_parquet
    from .logging_utils import progress, release_memory

    cfg = ctx.config
    clips = da.clips(ctx, split)
    clips = clips[clips["audio_member"].notna()]
    cache = ctx.paths.cache_dir("embeddings") / f"{split}_{extractor.alias}"
    cache.mkdir(parents=True, exist_ok=True)
    done = set()
    for p in cache.glob("chunk_*.parquet"):
        done |= set(pd.read_parquet(p, columns=["clip_id"])["clip_id"])
    todo = clips[~clips["clip_id"].isin(done)]
    archive = ctx.data_root() / ("train_audio.tar" if split == "train" else "test_audio.tar")
    extracted = ctx.paths.extracted_train if split == "train" else ctx.paths.extracted_test
    sr = 16000
    bs0 = int(extractor.spec.get("batch_size", cfg["audio_batch_size"]))
    chunk = max(bs0 * 8, 64)
    start = len(list(cache.glob("chunk_*.parquet")))
    for ci, s0 in enumerate(progress(range(0, len(todo), chunk), desc=f"embed {extractor.alias}")):
        part = todo.iloc[s0:s0 + chunk]
        use_files = all((extracted / m).exists() for m in part["audio_member"])
        blobs = {} if use_files else am.read_member_bytes(archive, part["audio_member"].tolist())
        vad = _vad_masks(ctx, part["clip_id"].tolist(), split)
        waves, ids = [], []
        for _, r in part.iterrows():
            src = str(extracted / r["audio_member"]) if use_files else blobs.get(r["audio_member"])
            if src is None:
                continue
            y, _ = decode_audio(src, sr)
            waves.append(y[: int(float(cfg["embedding_max_seconds"]) * sr)])
            ids.append(r["clip_id"])
        rows, k, bs = [], 0, bs0
        while k < len(waves):
            try:
                res = extractor.embed(waves[k:k + bs], sr, [vad.get(c) for c in ids[k:k + bs]])
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and bs > 1:
                    release_memory()
                    bs = max(1, bs // 2)
                    ctx.logger.warning("CUDA OOM during embedding; batch -> %d", bs)
                    continue
                raise
            for cid, rec in zip(ids[k:k + bs], res):
                row = {"clip_id": cid}
                for layer, pools in rec.items():
                    for pool, vec in pools.items():
                        row.update({f"emb_{extractor.alias}_{layer}_{pool}_{j}": float(v) for j, v in enumerate(vec)})
                rows.append(row)
            k += bs
        atomic_write_parquet(pd.DataFrame(rows), cache / f"chunk_{start + ci:05d}.parquet")
        release_memory()
    parts = [pd.read_parquet(p) for p in sorted(cache.glob("chunk_*.parquet"))]
    return pd.concat(parts, ignore_index=True).drop_duplicates("clip_id", keep="last") if parts else pd.DataFrame(columns=["clip_id"])


def reduce_and_pool(ctx, emb_train: pd.DataFrame, emb_test: pd.DataFrame | None, alias: str) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Train-fitted PCA per (layer, pool) block, then task-conditioned and student-level pooling (e_ prefix)."""
    from sklearn.decomposition import PCA

    cfg = ctx.config
    n_comp = int(cfg["embedding_pca_components"])
    blocks = {}
    for c in emb_train.columns:
        if c.startswith(f"emb_{alias}_"):
            key = "_".join(c.split("_")[:-1])
            blocks.setdefault(key, []).append(c)
    reduced_tr, reduced_te, info = [emb_train[["clip_id"]]], [emb_test[["clip_id"]]] if emb_test is not None else None, {}
    for key, cols in blocks.items():
        Xtr = emb_train[cols].to_numpy(dtype=float)
        mu = np.nanmean(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), mu, Xtr)
        k = min(n_comp, Xtr.shape[1], max(1, Xtr.shape[0] - 1))
        pca = PCA(k, random_state=int(cfg["seed"])).fit(Xtr)
        name = key.replace(f"emb_{alias}_", f"e_{alias}_")
        reduced_tr.append(pd.DataFrame(pca.transform(Xtr), columns=[f"{name}_pc{j}" for j in range(k)]))
        info[key] = {"components": k, "explained": float(pca.explained_variance_ratio_.sum())}
        if emb_test is not None:
            Xte = emb_test[cols].to_numpy(dtype=float)
            Xte = np.where(np.isnan(Xte), mu, Xte)
            reduced_te.append(pd.DataFrame(pca.transform(Xte), columns=[f"{name}_pc{j}" for j in range(k)]))

    def pool(red: pd.DataFrame, split: str) -> pd.DataFrame:
        cm = da.clips(ctx, split)[["clip_id", "student_id", "task_family"]]
        d = red.merge(cm, on="clip_id", how="inner")
        feats = [c for c in d.columns if c.startswith("e_")]
        g = d.groupby("student_id")[feats]
        out = pd.concat([g.mean().add_suffix("__mean"), g.std(ddof=1).add_suffix("__std")], axis=1)
        for task, gt in d.groupby("task_family"):
            out = out.join(gt.groupby("student_id")[feats].mean().add_suffix(f"__task={task}__mean"))
        return out.reset_index()

    tr = pd.concat(reduced_tr, axis=1)
    te = pd.concat(reduced_te, axis=1) if emb_test is not None else None
    return pool(tr, "train"), (pool(te, "test") if te is not None else None), info


def embedding_similarity(ctx, red_student: pd.DataFrame) -> dict:
    cols = [c for c in red_student.columns if c.endswith("__mean") and "__task=" not in c]
    if len(cols) < 2:
        return {}
    X = red_student[cols].fillna(0.0).to_numpy()
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim = X @ X.T
    cov = np.cov(red_student[cols].fillna(0.0).to_numpy().T)
    eig = np.linalg.eigvalsh(cov)[::-1]
    pos = eig[eig > 0]
    eff_rank = np.nan
    if len(pos):
        p = pos / pos.sum()
        eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    return {"mean_cosine_similarity": float(sim[np.triu_indices_from(sim, 1)].mean()), "effective_rank": eff_rank}


def stage_foundation_models(ctx) -> dict:
    cfg = ctx.config
    specs = cfg.get("embedding_models") or []
    if not specs:
        ctx.save_json({"status": "NOT_CONFIGURED", "reason": "no entries in CONFIG['embedding_models']"}, "embedding_status")
        return {"outputs": ["embedding_status"], "status": "SKIPPED_DISABLED", "reason": "no embedding models configured"}
    status, train_parts, test_parts = [], [], []
    for spec in specs:
        try:
            ext = EmbeddingExtractor(spec, cfg)
        except Exception as exc:
            status.append({"alias": spec.get("alias"), "status": "UNAVAILABLE", "reason": str(exc)})
            continue
        emb_tr = extract_embeddings(ctx, ext, "train")
        emb_te = extract_embeddings(ctx, ext, "test") if cfg.get("allow_test_inference") else None
        st_tr, st_te, info = reduce_and_pool(ctx, emb_tr, emb_te, ext.alias)
        train_parts.append(st_tr.set_index("student_id"))
        if st_te is not None:
            test_parts.append(st_te.set_index("student_id"))
        status.append({"alias": ext.alias, "status": "OK", "n_clips": int(len(emb_tr)), "pca": info,
                       "similarity": embedding_similarity(ctx, st_tr)})
        del ext
    outputs = ["embedding_status"]
    if train_parts:
        tr = pd.concat(train_parts, axis=1).reset_index()
        ctx.save_df(tr, "embedding_student_features")
        outputs.append("embedding_student_features")
        if test_parts:
            ctx.save_df(pd.concat(test_parts, axis=1).reset_index(), "embedding_student_features_test")
            outputs.append("embedding_student_features_test")
        y = da.target(ctx)
        cols = [c for c in tr.columns if c.endswith("__mean") and "__task=" not in c][:200]
        from .stats_utils import safe_corr

        rhos = pd.Series({c: safe_corr(tr.set_index("student_id")[c], y, "spearman", 20)[0] for c in cols})
        fig, axes = viz.new_figure(1, 1)
        viz.hist_kde(axes[0, 0], rhos.to_numpy(), "Spearman rho of embedding PCs with outcome")
        ctx.figure(fig, "embedding_target_associations", "Foundation-model embedding associations",
                   "Do pretrained representations capture outcome-related information?", "embedding_student_features",
                   int(len(tr)), f"Max |rho|={rhos.abs().max():.3f}; controlled comparison in Section 20 (EMBEDDINGS family).")
    ctx.save_json({"models": status}, "embedding_status")
    return {"outputs": outputs, "summary": {"models": status}}

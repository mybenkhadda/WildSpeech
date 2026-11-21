"""Reference-conditioned deviation features and the nested ReferenceProvider.

Per clip, relative to the exemplars of its item (exemplars never come from the clip's own student):
  dtw_mfcc_mean / dtw_mfcc_min      duration-normalised DTW cost (CMVN-MFCC) to the exemplars
  dtw_mel_mean                      same on mean-normalised log-mel
  local_dev_max / local_dev_pos     localized deviation (max moving-window cost) and where it starts
  warp_nondiag_frac / warp_diag_dev reference-conditioned temporal consistency (tempo irregularity, global warp)
  dur_ratio                         speech-span length / median exemplar length
  post_sim                          cosine of phoneme-posterior summaries to the exemplar centroid (P18, if any)
  emb_sim                           cosine of the pooled embedding to the exemplar centroid (P20, if any)
  ctc_expected_ll                   CTC expected-text likelihood passed through from P18 (if any)
Pairwise DTW results are cached on disk (fold-independent), so overlapping exemplar sets cost nothing twice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..features.conditional_aggregation import FoldFeatureProvider, ItemConditionedResidualizer, aggregate_student_task, clip_frame
from ..io_utils import atomic_write_parquet, stable_hash
from ..v2.validation_registry import inner_student_folds
from .dtw import dtw_batch_torch, dtw_numpy
from .templates import ExemplarBank

REF_FEATURES = ("dtw_mfcc_mean", "dtw_mfcc_min", "dtw_mel_mean", "local_dev_max", "local_dev_pos", "warp_nondiag_frac",
                "warp_diag_dev", "dur_ratio", "post_sim", "emb_sim", "ctc_expected_ll")


class PairDistanceCache:
    COLS = ("cost", "path_len", "norm_cost", "local_max", "local_max_pos", "nondiag_frac", "diag_dev")

    def __init__(self, root, fingerprint: str):
        self.dir = root / fingerprint
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mem: dict[tuple[str, str, str], tuple] = {}
        self.new: list[dict] = []
        for p in sorted(self.dir.glob("pairs-*.parquet")):
            df = pd.read_parquet(p)
            for r in df.itertuples(index=False):
                self.mem[(r.kind, r.a, r.b)] = tuple(getattr(r, c) for c in self.COLS)

    def get(self, kind, a, b):
        return self.mem.get((kind, a, b))

    def put(self, kind, a, b, d: dict) -> None:
        self.mem[(kind, a, b)] = tuple(d[c] for c in self.COLS)
        self.new.append({"kind": kind, "a": a, "b": b, **{c: d[c] for c in self.COLS}})

    def flush(self) -> None:
        if self.new:
            n = len(list(self.dir.glob("pairs-*.parquet")))
            atomic_write_parquet(pd.DataFrame(self.new), self.dir / f"pairs-{n:05d}.parquet")
            self.new = []


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


class ReferenceDeviationComputer:
    def __init__(self, ctx, sequences: dict[str, dict[str, np.ndarray]], cache: PairDistanceCache,
                 posteriors: pd.DataFrame | None = None, embeddings: pd.DataFrame | None = None,
                 ctc_ll: pd.Series | None = None):
        self.ctx = ctx
        r = ctx.cfg.get_path("v2.reference", {}) or {}
        self.seq, self.cache = sequences, cache
        self.band = r.get("band_frac", 0.35)
        self.window = int(r.get("local_window", 10))
        self.bs = int(r.get("dtw_batch_pairs", 384))
        self.device = ctx.device
        self.post = posteriors.set_index(FILE) if posteriors is not None and not posteriors.empty else None
        self.emb = embeddings.set_index(FILE) if embeddings is not None and not embeddings.empty else None
        self.ctc_ll = ctc_ll

    def _ensure_pairs(self, pairs: list[tuple[str, str, str]]) -> None:
        todo = [p for p in dict.fromkeys(pairs) if self.cache.get(*p) is None]
        for k in range(0, len(todo), self.bs):
            chunk = todo[k:k + self.bs]
            A = [self.seq[kind][a].astype(np.float32) for kind, a, b in chunk]
            B = [self.seq[kind][b].astype(np.float32) for kind, a, b in chunk]
            # group by kind (different dimensionality)
            for kind in sorted({p[0] for p in chunk}):
                idx = [i for i, p in enumerate(chunk) if p[0] == kind]
                try:        # vectorised wavefront DTW on the GPU, or on CPU torch when no GPU is present
                    res = dtw_batch_torch([A[i] for i in idx], [B[i] for i in idx], self.device, self.band, self.window)
                except ImportError:
                    res = [dtw_numpy(A[i], B[i], self.band, self.window) for i in idx]
                for i, d in zip(idx, res):
                    self.cache.put(*chunk[i], d)
        self.cache.flush()

    def compute(self, c: pd.DataFrame, bank: ExemplarBank) -> pd.DataFrame:
        """c: clip rows to score (FILE, ID, item_id). Returns FILE + REF features."""
        has = self.seq.get("mfcc", {})
        jobs, pairs = [], []
        for r in c[[FILE, ID, "item_id"]].itertuples(index=False):
            if r.filename not in has:
                continue
            ex = [e for e in bank.exemplars(r.item_id, exclude_student=str(r.subject_id)) if e in has and e != r.filename]
            if not ex:
                continue
            jobs.append((r.filename, ex))
            pairs += [(kind, r.filename, e) for e in ex for kind in ("mfcc", "mel")]
        self._ensure_pairs(pairs)
        rows = []
        for f, ex in jobs:
            d = {kind: np.array([self.cache.get(kind, f, e) for e in ex], float) for kind in ("mfcc", "mel")}
            m = d["mfcc"]
            ci = PairDistanceCache.COLS.index
            row = {FILE: f, "dtw_mfcc_mean": float(m[:, ci("norm_cost")].mean()), "dtw_mfcc_min": float(m[:, ci("norm_cost")].min()),
                   "dtw_mel_mean": float(d["mel"][:, ci("norm_cost")].mean()),
                   "local_dev_max": float(m[:, ci("local_max")].mean()), "local_dev_pos": float(m[:, ci("local_max_pos")].mean()),
                   "warp_nondiag_frac": float(m[:, ci("nondiag_frac")].mean()), "warp_diag_dev": float(m[:, ci("diag_dev")].mean()),
                   "dur_ratio": float(len(has[f]) / max(np.median([len(has[e]) for e in ex]), 1)), "n_exemplars": float(len(ex))}
            if self.post is not None and f in self.post.index:
                ex_p = [e for e in ex if e in self.post.index]
                if ex_p:
                    row["post_sim"] = _cos(self.post.loc[f].to_numpy(float), self.post.loc[ex_p].to_numpy(float).mean(0))
            if self.emb is not None and f in self.emb.index:
                ex_e = [e for e in ex if e in self.emb.index]
                if ex_e:
                    row["emb_sim"] = _cos(self.emb.loc[f].to_numpy(float), self.emb.loc[ex_e].to_numpy(float).mean(0))
            if self.ctc_ll is not None and f in self.ctc_ll.index:
                row["ctc_expected_ll"] = float(self.ctc_ll.loc[f])
            rows.append(row)
        out = pd.DataFrame(rows)
        for col in REF_FEATURES:
            if col not in out:
                out[col] = np.nan
        return out


class ReferenceProvider(FoldFeatureProvider):
    """Nested: clips of training students are scored against exemplar banks built from inner folds that exclude
    them; validation / test clips against the bank of all training students."""

    family = "REF"

    def __init__(self, ctx, name: str, computer: ReferenceDeviationComputer, fingerprint_extra: dict,
                 synthetic: dict[str, list[str]] | None = None, save_clip_dir=None):
        super().__init__(ctx, name, "reference-conditioned deviation")
        r = ctx.cfg.get_path("v2.reference", {}) or {}
        self.comp, self.extra, self.synthetic = computer, fingerprint_extra, synthetic or {}
        self.k, self.min_correct = int(r.get("exemplars_per_item", 4)), int(r.get("min_correct_per_item", 5))
        self.inner = int(r.get("inner_splits", 3))
        self.save_clip_dir = save_clip_dir

    def available(self) -> bool:
        return bool(self.comp.seq.get("mfcc"))

    def fingerprint_payload(self) -> dict:
        r = self.ctx.cfg.get_path("v2.reference", {}) or {}
        return {"extra": self.extra, "k": self.k, "min": self.min_correct, "inner": self.inner,
                "band": r.get("band_frac"), "win": r.get("local_window"), "syn": sorted(self.synthetic)}

    def clip_table(self, train_ids, target_ids) -> pd.DataFrame:
        c = clip_frame(self.ctx)
        avail = set(self.comp.seq.get("mfcc", {}))
        train_ids = sorted(map(str, train_ids))
        seed = int(int(stable_hash(train_ids), 16) % (2 ** 31 - 1))
        parts = []
        for in_tr, in_va in inner_student_folds(train_ids, self.inner, seed):
            bank = ExemplarBank(self.k, self.min_correct, self.ctx.seed).fit(c[c[ID].isin(in_tr)], avail, self.synthetic)
            parts.append(self.comp.compute(c[c[ID].isin(in_va)], bank).assign(source="inner"))
        others = sorted(set(map(str, target_ids)) - set(train_ids))
        if others:
            bank = ExemplarBank(self.k, self.min_correct, self.ctx.seed).fit(c[c[ID].isin(train_ids)], avail, self.synthetic)
            parts.append(self.comp.compute(c[c[ID].isin(others)], bank).assign(source="outer"))
        return pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame(columns=[FILE, *REF_FEATURES])

    def fit_table(self, train_ids, target_ids) -> pd.DataFrame:
        clip = self.clip_table(train_ids, target_ids)
        if self.save_clip_dir is not None and not clip.empty:
            atomic_write_parquet(clip, self.save_clip_dir / f"{self.name}__{stable_hash(sorted(map(str, train_ids)))}.parquet")
        c = clip_frame(self.ctx)
        ids = set(map(str, train_ids)) | set(map(str, target_ids))
        c = c[c[ID].isin(ids)].merge(clip, on=FILE, how="left").reset_index(drop=True)
        cols = [f for f in REF_FEATURES if f in c and c[f].notna().any()]
        if not cols:
            return pd.DataFrame(index=pd.Index(sorted(ids), name=ID))
        F = c[cols].to_numpy(float)
        tr = c[ID].isin(set(map(str, train_ids))).to_numpy()
        res = ItemConditionedResidualizer(("item", "domain"), 10.0).fit(c[tr], F[tr])
        Z = res.transform(c, F)
        out = aggregate_student_task(c, Z, list(cols), self.family, ("mean", "sd", "correct_mean", "incorrect_mean"))
        raw = aggregate_student_task(c, F[:, [cols.index(x) for x in cols if x in ("dur_ratio", "warp_nondiag_frac")]],
                                     [x for x in cols if x in ("dur_ratio", "warp_nondiag_frac")], self.family + "RAW", ("mean",), False) \
            if any(x in cols for x in ("dur_ratio", "warp_nondiag_frac")) else pd.DataFrame()
        if not raw.empty:
            raw.columns = [col.replace(f"{self.family}RAW__", f"{self.family}__").replace("__mean", "__raw_mean") for col in raw.columns]
            out = out.join(raw, how="outer")
        return out.reindex(sorted(ids))

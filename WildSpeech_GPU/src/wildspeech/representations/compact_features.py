"""Budgeted compact representations + the V3 provider factory.

CompactReducer(method, budget) — fitted on the TRAINING FOLD only, target = training-fold baseline residual:
    select  impute/scale -> top-`budget` columns by |Spearman| with the residual
    pca     impute/scale -> PCA with `budget` components (components FOLLOW the budget; V2 capped them at
            family_components=10, which is why its 20/50/100 budgets were identical)
    pls     PLS regression on the residual, `budget` latent components (supervised, fold-internal)
    rrr     reduced-rank ridge: ridge fit of the residual on all columns, then rank-`budget` projection of the
            fitted coefficient space via SVD of the ridge-smoothed design; rank and ridge alpha chosen by inner CV
Never more than min(budget, n_columns, n_train - 2) dimensions.

build_v3_providers(ctx, ...) -> {key: CachedProvider} for the ASR01-ASR12 feature families from the P25/P26/P27
tables (absent families are simply missing and reported upstream as NOT_EXECUTED).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.preprocessing import TabularPreprocessor
from .. import v3

METHODS = ("select", "pca", "pls", "rrr")


class CompactReducer:
    def __init__(self, method: str = "pca", budget: int = 20, seed: int = 0, max_missing_frac: float = 0.6,
                 rank_grid=(2, 4, 8), alphas=(1.0, 10.0, 100.0, 1000.0)):
        if method not in METHODS:
            raise ValueError(f"unknown reducer {method}")
        self.method, self.budget, self.seed, self.mmf = method, int(budget), int(seed), float(max_missing_frac)
        self.rank_grid, self.alphas = tuple(rank_grid), tuple(alphas)

    def fit(self, X: pd.DataFrame, y_resid) -> "CompactReducer":
        y = np.asarray(y_resid, float)
        self.prep_ = TabularPreprocessor(max_missing_frac=self.mmf, add_indicators=False, scale=True).fit(X)
        Zf = self.prep_.transform(X)
        Z = Zf.to_numpy(float)
        self.in_cols_ = list(Zf.columns)
        k =int(max(0, min(self.budget, Z.shape[1], len(y) - 2)))
        self.k_ = k
        if k == 0:
            return self
        if self.method == "select":
            from ..features.compact_representation import spearman_vector

            s = np.abs(spearman_vector(Z, y))
            self.idx_ = np.argsort(-s, kind="mergesort")[:k]
            self.scores_ = pd.Series(s, index=self.in_cols_)
        elif self.method == "pca":
            from sklearn.decomposition import PCA

            self.pca_ = PCA(n_components=k, random_state=self.seed).fit(Z)
        elif self.method == "pls":
            from sklearn.cross_decomposition import PLSRegression

            self.pls_ = PLSRegression(n_components=k, scale=False).fit(Z, y)
        else:
            self._fit_rrr(Z, y)
        return self

    def _fit_rrr(self, Z: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold

        best, best_err = (self.alphas[0], min(self.rank_grid[0], self.k_)), np.inf
        kf = KFold(4, shuffle=True, random_state=self.seed)
        for a in self.alphas:
            for r in self.rank_grid:
                r = int(min(r, self.k_))
                err = 0.0
                for tr, va in kf.split(Z):
                    W = self._rrr_basis(Z[tr], y[tr], a, r)
                    m = Ridge(alpha=1.0).fit(Z[tr] @ W, y[tr])
                    err += float(np.mean((y[va] - m.predict(Z[va] @ W)) ** 2))
                if err < best_err:
                    best, best_err = (a, r), err
        self.alpha_, self.rank_ = best
        self.W_ = self._rrr_basis(Z, y, *best)

    @staticmethod
    def _rrr_basis(Z: np.ndarray, y: np.ndarray, alpha: float, rank: int) -> np.ndarray:
        """Ridge-smoothed directions: top right-singular vectors of Z weighted by the ridge coefficients."""
        from sklearn.linear_model import Ridge

        b = Ridge(alpha=alpha).fit(Z, y).coef_.ravel()
        M = Z * b[None, :]
        _, _, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
        return Vt[:rank].T

    def transform(self, X: pd.DataFrame, prefix: str = "CMP") -> pd.DataFrame:
        if self.k_ == 0:
            return pd.DataFrame(index=X.index)
        Z = self.prep_.transform(X).to_numpy(float)
        if self.method == "select":
            A = Z[:, self.idx_]
            names = [f"{prefix}__all__sel_{self.in_cols_[i]}__value" for i in self.idx_]
        elif self.method == "pca":
            A = self.pca_.transform(Z)
            names = [f"{prefix}__all__pc{j}__value" for j in range(A.shape[1])]
        elif self.method == "pls":
            A = self.pls_.transform(Z)
            names = [f"{prefix}__all__pls{j}__value" for j in range(A.shape[1])]
        else:
            A = Z @ self.W_
            names = [f"{prefix}__all__rrr{j}__value" for j in range(A.shape[1])]
        return pd.DataFrame(A.astype(np.float32), index=X.index, columns=names)


# ============================================================================== provider factory
FAMILY_FEATURES = {
    v3.FAM_WORD: ("w_wer", "w_completeness", "w_order", "w_exact", "w_empty", "w_len_ratio", "w_repetitions",
                  "w_self_corrections", "w_sub_rate", "w_del_rate", "w_ins_rate", "c_cer"),
    v3.FAM_PHON: ("ph_per", "ph_sub_rate", "ph_del_rate", "ph_ins_rate", "ph_consonant_err", "ph_vowel_err",
                  "ph_cluster_acc", "ph_cluster_reduction", "ph_initial_omission", "ph_final_omission", "ph_err_initial",
                  "ph_err_medial", "ph_err_final", "ph_order_lcs", "ph_near_correct", "ph_sub_distance_mean",
                  "ph_sub_voicing_rate", "ph_sub_place_rate", "ph_sub_manner_rate"),
    v3.FAM_CONF: ("cf_tok_lp_mean", "cf_tok_lp_min", "cf_tok_low_frac", "cf_tok_entropy", "cf_expected_ll",
                  "cf_expected_gap", "cf_gop_mean", "cf_gop_min", "cf_gop_p10", "cf_low_conf_frac", "cf_vowel_gop",
                  "cf_consonant_gop", "cf_cluster_gop", "cf_post_mean", "cf_frame_entropy", "cf_supported_frac"),
    v3.FAM_TIME: ("tm_onset_s", "tm_phone_dur_mean", "tm_phone_dur_cv", "tm_vowel_dur_mean", "tm_consonant_dur_mean",
                  "tm_gap_mean", "tm_gap_cv", "tm_n_pauses", "tm_pause_within_word", "tm_pause_between_words",
                  "tm_artic_rate", "tm_s_per_phone", "tm_s_per_syllable"),
    v3.FAM_TASK: ("bl_integrated", "bl_segmented", "bl_transition_cv", "bl_partial", "bl_multiword_attempt", "dl_exact",
                  "dl_excessive", "dl_retained", "dl_retained_initial", "dl_retained_final", "dl_manipulation",
                  "dl_len_diff", "sr_omission_rate", "sr_sequencing", "sr_completeness", "sr_first_half_err",
                  "sr_second_half_err"),
}
CONSISTENCY_FEATURES = ("ph_near_correct", "cf_gop_mean", "cf_tok_lp_mean", "tm_s_per_phone", "ph_cluster_acc")


def build_v3_providers(ctx, wide: pd.DataFrame, details: pd.DataFrame, models: list[str], primary: str | None,
                       cons: pd.DataFrame | None = None, emb: pd.DataFrame | None = None, residual_fn=None,
                       include: tuple[str, ...] | None = None,
                       enc_embs: dict[str, pd.DataFrame] | None = None) -> tuple[dict, pd.DataFrame]:
    """-> ({family: CachedProvider}, availability table). `primary` = model whose features feed single-model
    families (correct-only, consistency, confusion); others contribute through ASRW/ASRP/ASRC/... columns."""
    from ..features.conditional_aggregation import CachedProvider
    from ..discovery.correct_response import CorrectResponseProvider, pick_role_columns
    from ..discovery.asr_disagreement import DisagreementProvider
    from ..phonetics.builder import feature_columns
    from .embedding_pooling import MILProvider, PooledEmbeddingProvider
    from .student_aggregation import (ConfusionEmbeddingProvider, ConsistencyProvider, NestedSoftCorrectnessProvider,
                                      phonetic_family_provider)

    root = ctx.paths.v3_cache / "fold_providers"
    inc = set(include or v3.V3_FAMILIES)
    out, rows = {}, []

    def add(key, prov, note=""):
        ok = prov is not None and prov.available()
        rows.append({"family": key, "available": bool(ok), "note": note if ok else (note or "no usable columns")})
        if ok:
            out[key] = prov if isinstance(prov, CachedProvider) else CachedProvider(prov, root=root)

    for fam, names in FAMILY_FEATURES.items():
        if fam in inc:
            cols = [c for m in models for c in feature_columns(wide, m, names)]
            add(fam, phonetic_family_provider(ctx, fam, wide, cols, name=f"{fam}_{'+'.join(models)}"), f"{len(cols)} clip columns")
    if primary:
        if v3.FAM_CORR in inc:
            roles = pick_role_columns(wide, primary)
            add(v3.FAM_CORR, CorrectResponseProvider(ctx, f"ASRO_{primary}", wide, roles, primary), f"roles={roles}")
        if v3.FAM_CONS in inc:
            cols = feature_columns(wide, primary, CONSISTENCY_FEATURES)
            add(v3.FAM_CONS, ConsistencyProvider(ctx, f"ASRS_{primary}", wide, cols), f"{len(cols)} columns")
        if v3.FAM_SOFT in inc:
            ns = ctx.cfg.get_path("v3.representations.nested_soft", {}) or {}
            cols = [c for f in (v3.FAM_WORD, v3.FAM_PHON, v3.FAM_CONF) for c in feature_columns(wide, primary, FAMILY_FEATURES[f])]
            add(v3.FAM_SOFT, NestedSoftCorrectnessProvider(ctx, f"ASRN_{primary}", wide, cols, ns.get("model", "LOGISTIC"),
                                                           int(ns.get("inner_splits", 4))), f"{len(cols)} clip columns")
        if v3.FAM_CONFU in inc and details is not None and not details.empty:
            add(v3.FAM_CONFU, ConfusionEmbeddingProvider(ctx, f"ASRX_{primary}", details, primary))
    if v3.FAM_DIS in inc:
        if cons is not None and not cons.empty:
            add(v3.FAM_DIS, DisagreementProvider(ctx, "ASRD", cons, float(ctx.cfg.get_path("v3.disagreement.high_uncertainty_quantile", 0.85))))
        else:
            rows.append({"family": v3.FAM_DIS, "available": False, "note": "fewer than two ASR models with outputs"})
    if v3.FAM_EMB in inc:
        if emb is not None and not emb.empty:
            rp = ctx.cfg.get_path("v3.representations", {}) or {}
            add(v3.FAM_EMB, PooledEmbeddingProvider(ctx, "ASRE_pooled", emb, 16))
            if residual_fn is not None and (rp.get("mil", {}) or {}).get("enabled", False):
                add("ASRE_MIL", MILProvider(ctx, "ASRE_mil", emb, residual_fn, rp.get("mil", {}),
                                            int((rp.get("mil", {}) or {}).get("embedding_pcs", 32))))
        else:
            rows.append({"family": v3.FAM_EMB, "available": False, "note": "no saved embeddings"})
    if v3.FAM_ENC in inc:
        encs = {k: e for k, e in (enc_embs or {}).items() if e is not None and not e.empty}
        if encs:
            class EncoderEmbeddingProvider(PooledEmbeddingProvider):
                family = v3.FAM_ENC

            npc = int((ctx.cfg.get_path("v3.representations", {}) or {}).get("encoder_pcs", 16))
            first = next(iter(encs))                                    # registry-priority order (providers.py)
            add(v3.FAM_ENC, EncoderEmbeddingProvider(ctx, f"ASRF_{first}", encs[first], npc), f"primary encoder {first}")
            for k, e in encs.items():
                add(f"ASRF_{k}", EncoderEmbeddingProvider(ctx, f"ASRF_{k}", e, npc), f"encoder {k}")
        else:
            rows.append({"family": v3.FAM_ENC, "available": False, "note": "no accepted frozen-encoder embeddings"})
    return out, pd.DataFrame(rows)

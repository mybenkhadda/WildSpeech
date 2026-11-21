"""StudentFeatureBuilder — clip-level data -> one row per student, with every learned quantity fitted on
training-fold students only.

Fitted in .fit(train_ids):
  * item p-correct (shrunk toward task mean; leave-one-student-out for training rows), item difficulty
  * item reference mean/std of every clip-level acoustic/phonetic feature (shrunk toward task level)
  * item log-duration reference
  * Rasch item parameters per task (held-out students scored from their own responses)
  * PCA of foundation-model embeddings (train-fold clips)
Computed in .transform(ids): item-normalised clip features aggregated per student x task (+ global):
  mean/std/median/p10/p90 (core), mean/std (others), correct-vs-incorrect contrast and difficulty slope
  (both shrunk), behaviour, difficulty-adjusted accuracy, IRT theta, linguistic load, screening, recording.

Column naming: FAMILY__scope__feature__stat (scope = task name or 'all'); FAMILY = text before the first '__'.
Recordings are never treated as independent observations: every output row is a student.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..irt import IRTModel

NON_AGG_META = {"META"}


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def family_of(col: str) -> str:
    return col.split("__", 1)[0]


def families_in(columns) -> list[str]:
    return sorted({family_of(c) for c in columns})


def select_families(columns, families: list[str] | None) -> list[str]:
    if not families or "ALL" in families:
        return list(columns)
    fam = set(families)
    return [c for c in columns if family_of(c) in fam]


class StudentFeatureBuilder:
    def __init__(self, clips: pd.DataFrame, students: pd.DataFrame, cfg, theta_cols: list[str],
                 clip_features: pd.DataFrame | None = None, text_table: pd.DataFrame | None = None,
                 instruments: list[str] | None = None):
        f = cfg.section("features")
        self.k = float(f.get("shrinkage_k", 10.0))
        self.k_slope = float(f.get("slope_shrinkage_k", 5.0))
        self.k_con = float(f.get("contrast_shrinkage_k", 3.0))
        self.core = {k.upper(): list(v) for k, v in (f.get("core_features") or {}).items()}
        self.core_stats = list(f.get("core_stats", ["mean", "std", "median", "p10", "p90"]))
        self.other_stats = list(f.get("other_stats", ["mean", "std"]))
        self.include_contrast = bool(f.get("include_contrast", True))
        self.include_slope = bool(f.get("include_slope", True))
        self.include_global = bool(f.get("include_global", True))
        self.include_irt = bool(f.get("include_irt", True))
        self.pca_k = int(cfg.get_path("embeddings.pca_components", 16))
        self.theta_cols = list(theta_cols)
        self.students = students.drop_duplicates(ID).set_index(ID)
        self.instruments = sorted(instruments or self.students.get("instrument", pd.Series(dtype=str)).dropna().unique().tolist())
        c = clips.copy()
        c["log_dur"] = np.log(c["duration_sec"].clip(lower=0.05))
        if clip_features is not None and not clip_features.empty:
            num = [col for col in clip_features.columns if col != FILE and "::" in col
                   and pd.api.types.is_numeric_dtype(clip_features[col]) and col.split("::")[0] not in NON_AGG_META]
            c = c.merge(clip_features[[FILE] + num], on=FILE, how="left")
        else:
            num = []
        self.emb_cols = [col for col in num if col.startswith("EMB_")]
        self.feat_cols = [col for col in num if not col.startswith("EMB_")]
        if text_table is not None and not text_table.empty:
            c = c.merge(text_table.add_prefix("TXT::").rename(columns={"TXT::item_id": "item_id"}), on="item_id", how="left")
        self.txt_cols = [col for col in c.columns if col.startswith("TXT::")]
        self.clips = c.reset_index(drop=True)
        self.tasks = sorted(self.clips["task"].unique())
        self.fitted_ = False

    # ================================================================= fit
    def fit(self, train_ids) -> "StudentFeatureBuilder":
        tr_ids = set(map(str, train_ids))
        self.train_ids_ = tr_ids
        tr = self.clips[self.clips[ID].isin(tr_ids)]
        sc = tr[tr["item_score"].notna()]
        self.task_p_ = sc.groupby("task")["item_score"].mean()
        g = sc.groupby("item_id")["item_score"]
        self.item_sum_, self.item_n_ = g.sum(), g.count()
        self.item_task_ = tr.drop_duplicates("item_id").set_index("item_id")["task"]
        # item difficulty on the logit scale (for slopes / adaptivity); computed without LOO for item identity
        p_item = (self.item_sum_ + self.k * self.item_task_.reindex(self.item_sum_.index).map(self.task_p_)) / (self.item_n_ + self.k)
        self.item_d_ = -np.log(p_item.clip(0.01, 0.99) / (1 - p_item.clip(0.01, 0.99)))
        self.task_d_ = -np.log(self.task_p_.clip(0.01, 0.99) / (1 - self.task_p_.clip(0.01, 0.99)))
        self.task_d_median_ = self.item_d_.groupby(self.item_task_.reindex(self.item_d_.index)).median()
        # reference statistics for item normalisation (log duration + clip features)
        ref_cols = ["log_dur"] + self.feat_cols
        gi = tr.groupby("item_id")[ref_cols]
        gt = tr.groupby("task")[ref_cols]
        n_i, m_i, v_i = gi.count(), gi.mean(), gi.var(ddof=0)
        m_t, v_t = gt.mean(), gt.var(ddof=0)
        task_of = self.item_task_.reindex(m_i.index)
        mt_i = m_t.reindex(task_of.values).set_index(m_i.index)
        vt_i = v_t.reindex(task_of.values).set_index(m_i.index)
        w = n_i / (n_i + self.k)
        self.ref_mean_item_ = (w * m_i.fillna(mt_i) + (1 - w) * mt_i)
        self.ref_var_item_ = (w * v_i.fillna(vt_i) + (1 - w) * vt_i)
        self.ref_mean_task_, self.ref_var_task_ = m_t, v_t
        # IRT per task
        self.irt_ = {}
        if self.include_irt:
            for t in self.tasks:
                d = sc[sc["task"] == t]
                if d["item_id"].nunique() >= 3 and d[ID].nunique() >= 10:
                    self.irt_[t] = IRTModel("rasch").fit(d)
        # embedding PCA per model (train-fold clips only)
        self.pca_ = {}
        if self.emb_cols:
            from sklearn.decomposition import PCA

            for model in sorted({c.split("::")[0] for c in self.emb_cols}):
                cols = [c for c in self.emb_cols if c.startswith(model + "::")]
                X = tr[cols].to_numpy(np.float32)
                ok = np.isfinite(X).all(1)
                if ok.sum() < max(20, self.pca_k + 2):
                    continue
                mu, sd = X[ok].mean(0), X[ok].std(0) + 1e-6
                pca = PCA(n_components=min(self.pca_k, ok.sum() - 1, len(cols)), random_state=0).fit((X[ok] - mu) / sd)
                self.pca_[model] = (cols, mu, sd, pca)
        self.fitted_ = True
        return self

    # =========================================================== clip-level
    def _clip_frame(self, ids, clip_mask: pd.Series | None = None) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("call fit() first")
        ids = set(map(str, ids))
        m = self.clips[ID].isin(ids)
        if clip_mask is not None:
            m &= clip_mask.reindex(self.clips.index).fillna(False).astype(bool)
        c = self.clips[m].copy()
        task_p = c["task"].map(self.task_p_).fillna(self.task_p_.mean() if len(self.task_p_) else 0.5)
        S = c["item_id"].map(self.item_sum_).fillna(0.0)
        N = c["item_id"].map(self.item_n_).fillna(0.0)
        # leave-one-student-out correction for training students
        in_train = c[ID].isin(self.train_ids_)
        own = c.assign(_s=c["item_score"]).groupby([ID, "item_id"])["_s"].agg(["sum", "count"])
        key = pd.MultiIndex.from_arrays([c[ID], c["item_id"]])
        own_s = pd.Series(own["sum"].reindex(key).to_numpy(), index=c.index).where(in_train, 0.0).fillna(0.0)
        own_n = pd.Series(own["count"].reindex(key).to_numpy(), index=c.index).where(in_train, 0.0).fillna(0.0)
        c["p_item"] = ((S - own_s) + self.k * task_p) / ((N - own_n) + self.k)
        c["d_item"] = c["item_id"].map(self.item_d_)
        c["d_item"] = c["d_item"].fillna(c["task"].map(self.task_d_)).fillna(0.0)
        c["resid_correct"] = c["item_score"] - c["p_item"]
        c["hard_item"] = (c["d_item"] > c["task"].map(self.task_d_median_).fillna(0.0)).astype(float)
        ref_cols = ["log_dur"] + self.feat_cols
        items, tasks = c["item_id"].to_numpy(), c["task"].to_numpy()
        mu = self.ref_mean_item_.reindex(items)[ref_cols].to_numpy(float)
        var = self.ref_var_item_.reindex(items)[ref_cols].to_numpy(float)
        mu = np.where(np.isnan(mu), self.ref_mean_task_.reindex(tasks)[ref_cols].to_numpy(float), mu)
        var = np.where(np.isnan(var), self.ref_var_task_.reindex(tasks)[ref_cols].to_numpy(float), var)
        Z = (c[ref_cols].to_numpy(float) - mu) / np.sqrt(var + 1e-12)
        Z = np.where(np.isfinite(Z), np.clip(Z, -8, 8), np.nan)
        zdf = pd.DataFrame(Z, columns=["duration_z"] + self.feat_cols, index=c.index)
        c = c.drop(columns=self.feat_cols).join(zdf)
        for model, (cols, mu_e, sd_e, pca) in self.pca_.items():
            X = c[cols].to_numpy(np.float32)
            ok = np.isfinite(X).all(1)
            P = np.full((len(c), pca.n_components_), np.nan, dtype=np.float32)
            if ok.any():
                P[ok] = pca.transform((X[ok] - mu_e) / sd_e)
            c = c.drop(columns=cols)
            for j in range(P.shape[1]):
                c[f"{model}::pc{j}"] = P[:, j]
        leftover = [col for col in self.emb_cols if col in c.columns]   # models without a fitted PCA
        return c.drop(columns=leftover)

    def clip_level(self, ids, families: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
        """Item-normalised clip matrix for MIL / attention models (+ item difficulty & correctness)."""
        c = self._clip_frame(ids)
        fam_cols = [col for col in c.columns if "::" in col and not col.startswith("TXT::")]
        if families and "ALL" not in families:
            fam_cols = [col for col in fam_cols if col.split("::")[0] in set(families)]
        cols = ["d_item", "resid_correct", "is_scored", "duration_z", "hard_item"] + self.txt_cols + fam_cols
        out = c[[ID, "task", "item_id", "item_score"] + cols].copy()
        out["resid_correct"] = out["resid_correct"].fillna(0.0)
        return out, cols

    # ============================================================ aggregate
    @staticmethod
    def _wide(df: pd.DataFrame, fam: str, stat: str, ids_index) -> pd.DataFrame:
        """(ID, task)-indexed frame of features -> wide FAM__task__feat__stat."""
        if df.empty:
            return pd.DataFrame(index=ids_index)
        w = df.unstack("task")
        w.columns = [f"{fam}__{t}__{f}__{stat}" for f, t in w.columns]
        return w.reindex(ids_index)

    def _stats(self, c: pd.DataFrame, cols: list[str], fam: str, stats: list[str], idx, keys) -> list[pd.DataFrame]:
        out = []
        if not cols:
            return out
        g = c.groupby(keys)[cols]
        cnt = g.count()
        for s in stats:
            if s == "mean":
                v = g.mean()
            elif s == "std":
                v = g.std(ddof=0).where(cnt >= 2)
            elif s == "median":
                v = g.median()
            elif s in ("p10", "p90"):
                v = g.quantile(0.1 if s == "p10" else 0.9)
            else:
                continue
            if keys == [ID, "task"]:
                out.append(self._wide(v, fam, s, idx))
            else:
                v.columns = [f"{fam}__all__{f}__{s}" for f in v.columns]
                out.append(v.reindex(idx))
        return out

    def _contrast(self, c: pd.DataFrame, cols: list[str], fam: str, idx) -> pd.DataFrame:
        d = c[c["item_score"].notna()]
        if d.empty or not cols:
            return pd.DataFrame(index=idx)
        g = d.groupby([ID, "task", "item_score"])[cols]
        mean, cnt = g.mean(), g.count()
        try:
            m1, m0 = mean.xs(1.0, level="item_score"), mean.xs(0.0, level="item_score")
            n1, n0 = cnt.xs(1.0, level="item_score"), cnt.xs(0.0, level="item_score")
        except KeyError:
            return pd.DataFrame(index=idx)
        m1, m0 = m1.align(m0, join="outer")
        n1, n0 = n1.align(n0, join="outer")
        n1, n0 = n1.reindex(m1.index).fillna(0), n0.reindex(m1.index).fillna(0)
        neff = 2 * n1 * n0 / (n1 + n0).replace(0, np.nan)
        con = ((m1 - m0) * (neff / (neff + self.k_con))).fillna(0.0)
        return self._wide(con, fam, "contrast", idx)

    def _slope(self, c: pd.DataFrame, cols: list[str], fam: str, idx) -> pd.DataFrame:
        if not cols:
            return pd.DataFrame(index=idx)
        X = c[cols].to_numpy(float)
        M = np.isfinite(X).astype(float)
        X0 = np.nan_to_num(X)
        d = c["d_item"].to_numpy(float)[:, None]
        keys = pd.MultiIndex.from_arrays([c[ID].values, c["task"].values], names=[ID, "task"])
        parts = {"n": M, "sx": X0, "sd": d * M, "sxd": X0 * d, "sdd": d * d * M}
        sums = {k: pd.DataFrame(v, index=keys, columns=cols).groupby(level=[0, 1]).sum() for k, v in parts.items()}
        n = sums["n"]
        cov = sums["sxd"] - sums["sx"] * sums["sd"] / n.replace(0, np.nan)
        var = sums["sdd"] - sums["sd"] ** 2 / n.replace(0, np.nan)
        slope = (cov / var.where(var > 1e-9)) * (n / (n + self.k_slope))
        slope = slope.where(n >= 3).fillna(0.0)
        return self._wide(slope, fam, "slope", idx)

    def transform(self, ids, clip_mask: pd.Series | None = None) -> pd.DataFrame:
        ids = [str(i) for i in ids]
        idx = pd.Index(ids, name=ID)
        c = self._clip_frame(ids, clip_mask)
        keys = [ID, "task"]
        blocks: list[pd.DataFrame] = []

        # ---- BEHAVIOR
        b = c.assign(n_items=1.0, unscored=1.0 - c["is_scored"], correct_d=np.where(c["item_score"] == 1, c["d_item"], np.nan))
        gb = b.groupby(keys)
        beh = pd.DataFrame({"n_items": gb["n_items"].sum(), "n_scored": gb["is_scored"].sum(),
                            "frac_unscored": gb["unscored"].mean(), "accuracy": gb["item_score"].mean(),
                            "mean_difficulty": gb["d_item"].mean(), "max_difficulty_correct": gb["correct_d"].max(),
                            "mean_log_duration": gb["log_dur"].mean(), "total_duration": gb["duration_sec"].sum()})
        blocks.append(self._wide(beh, "BEHAVIOR", "value", idx))
        blocks += self._stats(c, ["duration_z"], "BEHAVIOR", ["mean", "std"], idx, keys)
        if self.include_slope:
            blocks.append(self._slope(c, ["duration_z"], "BEHAVIOR", idx))
        ga = b.groupby(ID)
        blocks.append(pd.DataFrame({"BEHAVIOR__all__n_items__value": ga["n_items"].sum(),
                                    "BEHAVIOR__all__frac_unscored__value": ga["unscored"].mean(),
                                    "BEHAVIOR__all__accuracy__value": ga["item_score"].mean(),
                                    "BEHAVIOR__all__total_duration__value": ga["duration_sec"].sum(),
                                    "BEHAVIOR__all__n_tasks__value": b.groupby(ID)["task"].nunique()}).reindex(idx))

        # ---- ITEM (difficulty-adjusted accuracy, hard/easy gap, IRT)
        gi = c.groupby(keys)
        item = pd.DataFrame({"adj_accuracy": gi["resid_correct"].mean()})
        hard = c[c["hard_item"] == 1].groupby(keys)["item_score"].mean()
        easy = c[c["hard_item"] == 0].groupby(keys)["item_score"].mean()
        item["hard_easy_gap"] = (hard - easy).reindex(item.index)
        w = self._wide(item, "ITEM", "value", idx)
        w.columns = [col.replace("__value", "") if "adj_accuracy" in col else col for col in w.columns]
        blocks.append(w)
        blocks.append(pd.DataFrame({"ITEM__all__adj_accuracy": c.groupby(ID)["resid_correct"].mean()}).reindex(idx))
        for t, model in getattr(self, "irt_", {}).items():
            s = model.score_persons(c[c["task"] == t]).set_index("subject_id")
            blocks.append(pd.DataFrame({f"ITEM__{t}__irt_theta__value": s["theta"], f"ITEM__{t}__irt_se__value": s["theta_se"]}).reindex(idx))

        # ---- acoustic / phonetic / embedding families
        fam_cols: dict[str, list[str]] = {}
        for col in c.columns:
            if "::" in col and not col.startswith("TXT::"):
                fam_cols.setdefault(col.split("::")[0], []).append(col)
        for fam, cols in sorted(fam_cols.items()):
            renamed = {col: col.split("::", 1)[1] for col in cols}
            cc = c[[ID, "task", "item_score", "d_item"] + cols].rename(columns=renamed)
            feats = list(renamed.values())
            core = [f for f in self.core.get(fam, []) if f in feats]
            other = [f for f in feats if f not in core]
            blocks += self._stats(cc, core, fam, self.core_stats, idx, keys)
            blocks += self._stats(cc, other, fam, self.other_stats, idx, keys)
            if self.include_contrast and core:
                blocks.append(self._contrast(cc, core, fam, idx))
            if self.include_slope and core:
                blocks.append(self._slope(cc, core, fam, idx))
            if self.include_global:
                blocks += self._stats(cc, core or feats[:20], fam, ["mean", "std"], idx, [ID])

        # ---- LINGUISTIC (load of administered items; adaptive-test signal) + speech per syllable
        if self.txt_cols:
            tc = c[[ID, "task"] + self.txt_cols].rename(columns={x: x.split("::", 1)[1] for x in self.txt_cols})
            lcols = [x for x in ("n_syllables", "n_phonemes", "n_words", "max_cluster", "mean_word_len") if x in tc.columns]
            blocks += self._stats(tc, lcols, "LINGUISTIC", ["mean"], idx, keys)
            raw_sp = self.clips.loc[c.index, "TEMPORAL::speech_dur_s"] if "TEMPORAL::speech_dur_s" in self.clips else None
            if raw_sp is not None and "TXT::n_syllables" in c:
                sps = (raw_sp / c["TXT::n_syllables"].replace(0, np.nan)).rename("speech_s_per_syllable")
                tmp = pd.concat([c[[ID, "task"]], sps], axis=1)
                blocks += self._stats(tmp, ["speech_s_per_syllable"], "LINGUISTIC", ["mean", "median"], idx, keys)

        # ---- SCREENING + RECORDING (student metadata available in both splits)
        st = self.students.reindex(idx)
        scr = pd.DataFrame(index=idx)
        for col in self.theta_cols:
            if col in st:
                scr[f"SCREENING__{col}"] = st[col].astype(float)
                scr[f"SCREENING__{col}_missing"] = st[col].isna().astype(float)
        if self.theta_cols:
            scr["SCREENING__theta_mean"] = st[[c_ for c_ in self.theta_cols if c_ in st]].astype(float).mean(axis=1)
        if "instrument" in st:
            for ins in self.instruments:
                scr[f"SCREENING__instrument_{ins}"] = (st["instrument"] == ins).astype(float)
        blocks.append(scr)
        if "original_sample_rate" in st:
            rec = pd.DataFrame({"RECORDING__orig_sr_khz": st["original_sample_rate"].astype(float) / 1000.0,
                                "RECORDING__orig_sr_low": (st["original_sample_rate"].astype(float) < 16000).astype(float)}, index=idx)
            blocks.append(rec)

        X = pd.concat([blk for blk in blocks if blk is not None and blk.shape[1]], axis=1)
        X.columns = [sanitize(col) for col in X.columns]
        X = X.loc[:, ~X.columns.duplicated()]
        return X.replace([np.inf, -np.inf], np.nan)


def crossfit_features(builder: StudentFeatureBuilder, ids: list[str], y: pd.Series, n_folds: int, seed: int) -> pd.DataFrame:
    """Out-of-fold student features: each student's row comes from a builder fitted without that student."""
    from ..validation.splits import stratified_folds

    parts = []
    for tr, va in stratified_folds(pd.Series(y.reindex(ids).values, index=ids), n_folds, seed):
        parts.append(builder.fit(tr).transform(va))
    X = pd.concat(parts, axis=0)
    return X.reindex(ids)


def item_parity_mask(clips: pd.DataFrame, parity: int) -> pd.Series:
    """Split-half reliability: alternate items within each student x task (by item order in metadata)."""
    rank = clips.groupby([ID, "task"]).cumcount()
    return (rank % 2 == parity)

"""P09 — experimental composite literacy indices (NOT validated educational diagnoses).

An index = unit-weighted mean of standardised components whose signs are learned INSIDE each training fold
(sign of the rank correlation with the training-fold screening residual). Evaluated by:
  * incremental CV R2 over the screening model (paired bootstrap CI),
  * internal consistency (Cronbach alpha on oriented components),
  * split-half reliability across alternating items (Spearman-Brown corrected).
"""
from __future__ import annotations

import fnmatch

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from ..features.builder import family_of
from ..stats import cronbach_alpha, spearman, spearman_brown

DISCLAIMER = ("Indices are experimental measurement constructs derived from anonymised assessment audio. "
              "They are not established educational or clinical diagnoses and must not be used to label students.")
ALPHAS = np.logspace(-2, 4, 31)


def resolve_components(columns, patterns: list[str]) -> list[str]:
    out = []
    for p in patterns:
        out += [c for c in columns if fnmatch.fnmatchcase(c, p) and c not in out]
    return out


class CompositeIndex:
    def __init__(self, name: str, components: list[str]):
        self.name, self.components = name, list(components)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CompositeIndex":
        scr = [c for c in X.columns if family_of(c) == "SCREENING"]
        Zs = X[scr].fillna(X[scr].median())
        base = RidgeCV(alphas=ALPHAS).fit(Zs, y.values)
        resid = pd.Series(y.values - base.predict(Zs), index=X.index)
        C = X[self.components]
        self.med_, self.mu_ = C.median(), C.fillna(C.median()).mean()
        self.sd_ = C.fillna(C.median()).std(ddof=0).replace(0, 1.0)
        signs = {}
        for c in self.components:
            r = spearman(C[c], resid)[0]
            signs[c] = float(np.sign(r)) if np.isfinite(r) and r != 0 else 0.0
        self.signs_ = pd.Series(signs)
        self.active_ = [c for c in self.components if self.signs_[c] != 0]
        return self

    def oriented(self, X: pd.DataFrame) -> pd.DataFrame:
        C = X.reindex(columns=self.active_)
        Z = (C.fillna(self.med_[self.active_]) - self.mu_[self.active_]) / self.sd_[self.active_]
        return Z * self.signs_[self.active_]

    def transform(self, X: pd.DataFrame) -> pd.Series:
        if not self.active_:
            return pd.Series(np.nan, index=X.index, name=self.name)
        return self.oriented(X).mean(axis=1).rename(self.name)


def evaluate_index_cv(runner, name: str, patterns: list[str], scheme: str = "repeated_kfold") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outer-CV: per fold build features, fit index on train, evaluate screening vs screening+index ridge."""
    rows, comp_rows = [], []
    for f in runner.folds(scheme):
        Xtr, Xva = runner.fold_data(f)
        comps = resolve_components(Xtr.columns, patterns)
        if len(comps) < 2:
            continue
        ytr = runner.y.reindex(Xtr.index)
        idx = CompositeIndex(name, comps).fit(Xtr, ytr)
        scr = [c for c in Xtr.columns if family_of(c) == "SCREENING"]
        med = Xtr[scr].median()
        A_tr, A_va = Xtr[scr].fillna(med), Xva[scr].fillna(med)
        m0 = RidgeCV(alphas=ALPHAS).fit(A_tr, ytr.values)
        it, iv = idx.transform(Xtr), idx.transform(Xva)
        B_tr, B_va = A_tr.assign(**{name: it}), A_va.assign(**{name: iv})
        m1 = RidgeCV(alphas=ALPHAS).fit(B_tr, ytr.values)
        yva = runner.y.reindex(Xva.index).values
        rows.append(pd.DataFrame({"subject_id": Xva.index, "repeat": f.repeat, "fold": f.fold, "y_true": yva,
                                  "pred_base": m0.predict(A_va), "pred_index": m1.predict(B_va), "index_value": iv.values}))
        comp_rows.append({"repeat": f.repeat, "fold": f.fold, "n_components": len(idx.active_),
                          "alpha": cronbach_alpha(idx.oriented(Xtr).values), **{f"sign__{c}": idx.signs_[c] for c in comps}})
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), pd.DataFrame(comp_rows)


def split_half_reliability(builder, train_ids: list[str], y: pd.Series, name: str, patterns: list[str]) -> dict:
    """Fit features on all train ids, build each student's index from odd vs even items, correlate."""
    from ..features.builder import item_parity_mask

    builder.fit(train_ids)
    clips = builder.clips
    halves = []
    for parity in (0, 1):
        Xh = builder.transform(train_ids, clip_mask=item_parity_mask(clips, parity))
        halves.append(Xh)
    comps = resolve_components(halves[0].columns, patterns)
    if len(comps) < 2:
        return {"index": name, "split_half_r": np.nan, "spearman_brown": np.nan, "n_components": len(comps)}
    full = builder.transform(train_ids)
    idx = CompositeIndex(name, comps).fit(full, y.reindex(full.index))
    a, b = idx.transform(halves[0]), idx.transform(halves[1])
    r = spearman(a, b)[0]
    return {"index": name, "split_half_r": r, "spearman_brown": spearman_brown(r), "n_components": len(idx.active_)}

"""OOF metrics, paired comparisons and conservative preregistered promotion gates."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _r2(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all() or np.var(a) <= 1e-12:
        return float("nan")
    return float(r2_score(a, b))


def _scores(y, p):
    return {"r2": _r2(y, p), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p)))}


def validate_oof(oof: pd.DataFrame, folds: pd.DataFrame, ids, y):
    required = {"scheme","repeat","fold","pos","subject_id","y_true","y_pred"}
    if required - set(oof):
        raise ValueError(f"OOF missing fields: {required - set(oof)}")
    n = len(ids)
    if len(y) != n or not np.isfinite(oof[["y_true","y_pred"]].to_numpy(dtype=float)).all():
        raise ValueError("Invalid targets or predictions")
    for row in folds.itertuples():
        sub = oof[(oof.scheme==row.scheme) & (oof.repeat==row.repeat) & (oof.fold==row.fold)]
        positions = sub.pos.to_numpy(int)
        if (len(sub)!=len(row.valid_pos) or len(set(positions)) != len(positions)
                or set(positions)!=set(map(int,row.valid_pos))):
            raise ValueError(f"OOF coverage mismatch in {row.scheme}/{row.repeat}/{row.fold}")
        if any(ids[p] != sid for p,sid in zip(positions,sub.subject_id)):
            raise ValueError("OOF subject/position mismatch")
        if not np.allclose(sub.y_true.to_numpy(float),np.asarray(y)[positions],rtol=0,atol=0):
            raise ValueError("OOF label alignment failed")
    if len(oof)!=int(folds.n_valid.sum()):
        raise ValueError("Unexpected extra OOF predictions")
    if oof.duplicated(["scheme","repeat","pos"]).any():
        raise ValueError("A student appears twice in one repeat")
    return True


def summarize_oof(oof:pd.DataFrame, folds:pd.DataFrame, ids, y,
                  instruments=None, min_instrument_n=20):
    validate_oof(oof,folds,ids,y)
    rows=[]
    for (scheme,repeat,fold),g in oof.groupby(["scheme","repeat","fold"],sort=True):
        rows.append(dict(scheme=scheme,repeat=int(repeat),fold=int(fold),
                         n=len(g),**_scores(g.y_true,g.y_pred)))
    fold_metrics=pd.DataFrame(rows).sort_values(["scheme","repeat","fold"]).reset_index(drop=True)
    pooled=[]
    for (scheme,rep),g in oof.groupby(["scheme","repeat"]):
        if len(g)!=len(ids):
            raise ValueError("Each OOF repeat must have exactly one prediction per student")
        sc=_scores(g.y_true,g.y_pred)
        pooled.append(dict(scheme=scheme,repeat=int(rep),n=len(g),**sc))
    pooled=pd.DataFrame(pooled).sort_values(["scheme","repeat"]).reset_index(drop=True)
    subgroups=[]
    if instruments is not None:
        instruments=np.asarray(instruments,dtype=str)
        for (scheme,rep),g in oof.groupby(["scheme","repeat"]):
            d=g.assign(instrument=instruments[g.pos.to_numpy(int)])
            for instrument,x in d.groupby("instrument"):
                val = _r2(x.y_true,x.y_pred) if len(x)>=min_instrument_n else float("nan")
                subgroups.append({"scheme":scheme,"repeat":int(rep),"instrument":str(instrument),
                                  "n":len(x),"r2":val,
                                  "variance":float(x.y_true.var(ddof=0)),
                                  "sse":float(np.square(x.y_true-x.y_pred).sum())})
    instrument_metrics=pd.DataFrame(subgroups)
    return fold_metrics,pooled,instrument_metrics


def corrected_t(delta_r2, n_folds=5):
    """Approximate Nadeau-Bengio repeated-CV corrected t (overlap-sensitive)."""
    x=np.asarray(delta_r2,dtype=float)
    if len(x)<2 or not np.isfinite(x).all():
        return float("nan")
    mean=float(np.mean(x))
    sd=float(np.std(x,ddof=1))
    if sd < 1e-15:
        return float("inf") if mean>0 else (float("-inf") if mean<0 else 0.)
    return mean/(sd*math.sqrt(1./len(x) + 1./(n_folds-1)))


def compare_runs(base_oof,challenger_oof,folds,ids,y,instruments,
                 base_fold_hash, challenger_fold_hash,
                 baseline_name="REF_RIDGE14",challenger_name="CANDIDATE"):
    if base_fold_hash != challenger_fold_hash:
        raise ValueError("FOLD HASH MISMATCH: paired comparison prohibited")
    validate_oof(base_oof,folds,ids,y)
    validate_oof(challenger_oof,folds,ids,y)
    keys=["scheme","repeat","fold","pos"]
    a=base_oof.set_index(keys).sort_index()
    b=challenger_oof.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("OOF fold/student alignment mismatch")
    if not (a.subject_id.equals(b.subject_id) and a.y_true.equals(b.y_true)):
        raise ValueError("OOF subject or target mismatch")
    af,ap,ai=summarize_oof(base_oof,folds,ids,y,instruments)
    bf,bp,bi=summarize_oof(challenger_oof,folds,ids,y,instruments)
    k=["scheme","repeat","fold"]
    aligned=af.merge(bf,on=k,suffixes=("_base","_new"),validate="one_to_one")
    aligned["delta_r2"]=aligned.r2_new-aligned.r2_base
    student=aligned[aligned.scheme=="student"]
    if len(student)!=50 or not np.isfinite(student.delta_r2).all():
        raise ValueError("Exactly 50 finite paired student-fold R² deltas required")
    pooled=ap.merge(bp,on=["scheme","repeat"],suffixes=("_base","_new"),validate="one_to_one")
    pooled["delta_r2"]=pooled.r2_new-pooled.r2_base
    sch=pooled[pooled.scheme=="school"]
    inst=ai.merge(bi,on=["scheme","repeat","instrument"],suffixes=("_base","_new"),validate="one_to_one")
    inst["delta_r2"]=inst.r2_new-inst.r2_base
    inst=inst[inst.scheme=="student"]
    instrument_delta={}
    for instrument,g in inst.groupby("instrument"):
        vals=g.delta_r2.to_numpy(float)
        instrument_delta[str(instrument)] = float(np.mean(vals)) if np.isfinite(vals).all() else None
    d=student.delta_r2.to_numpy(float)
    delta_school=float(sch.delta_r2.mean()) if len(sch)==10 and np.isfinite(sch.delta_r2).all() else None
    return {
        "baseline":baseline_name,"challenger":challenger_name,
        "mean_fold_delta_r2":float(np.mean(d)),
        "sd_fold_delta_r2":float(np.std(d,ddof=1)),
        "corrected_t":corrected_t(d,5),
        "win_fraction":float(np.mean(d>0)),
        "tie_fraction":float(np.mean(d==0)),
        "n_paired_folds":len(d),
        "mean_student_pooled_delta_r2":float(pooled[pooled.scheme=="student"].delta_r2.mean()),
        "school_pooled_delta_r2":delta_school,
        "instrument_delta_r2":instrument_delta,
        "paired_fold_data":aligned[["scheme","repeat","fold","delta_r2"]].to_dict("records")
    }


def promotion_gate(cmp, min_delta=.010, min_t=2., min_win=.6,
                   max_school_drop=-.005, max_instrument_drop=-.01):
    d=cmp["mean_fold_delta_r2"]
    s=cmp.get("school_pooled_delta_r2")
    inst=cmp.get("instrument_delta_r2",{})
    if not np.isfinite(d):
        return "DROP", "Invalid or nonfinite paired improvement"
    if s is None or not np.isfinite(s) or not inst or any(v is None for v in inst.values()):
        return "DROP", "Missing mandatory school/instrument robustness evidence"
    if s < max_school_drop or any(v < max_instrument_drop for v in inst.values()):
        return "DROP", "Failed predeclared generalization veto"
    if d <= 0:
        return "DROP", "No positive average fold R² improvement"
    t=cmp["corrected_t"]
    win=cmp["win_fraction"]
    if d>=min_delta and t>=min_t and win>=min_win:
        return "PROMOTE", "All improvement and robustness gates satisfied"
    return "HOLD", "Positive improvement but below one or more promotion thresholds"

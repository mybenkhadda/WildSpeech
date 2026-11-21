"""P20 — speech foundation-model discovery (eligible local checkpoints only).

GPU extraction of every hidden layer with mean / std / speech-masked pooling (fp16, duration-aware batches,
OOM recovery, float16 shards, resume). Layer discovery on a fixed student-grouped clip sample: correctness
decodability (PCA-logistic and supervised PLS), phonetic-content preservation (kNN item identity among correct
responses), recording-domain leakage, layer CKA similarity. Pre-registered selection of a few layers ->
student-level compact representation (fold-fitted PCA, nested item-centroid similarity, nested supervised
bottleneck) -> incremental R2 over the strongest tabular comparator -> selection.json for P17/P19/P21.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..features.conditional_aggregation import CachedProvider, clip_frame
from ..foundation.compression import FoundationProvider, LayerEvaluator, linear_cka
from ..foundation.encoders import EmbeddingV2Runner, embedding_jobs, open_store
from ..models.compact_regression import CompactSpec, StudentLevelEvaluator
from ..v2 import visualization_v2 as V
from ..v2.registry import save_figure_v2
from ._v2common import begin, evaluate_against, schemes

S = "P20"


def _sample_students(c: pd.DataFrame, n_clips: int, seed: int) -> pd.DataFrame:
    ids = pd.Series(sorted(c[ID].astype(str).unique()))
    order = ids.iloc[np.argsort(pd.util.hash_pandas_object(ids + f"#{seed}", index=False).to_numpy(), kind="mergesort")]
    cnt = c.groupby(ID).size().reindex(order).fillna(0).cumsum()
    keep = set(cnt[cnt <= n_clips].index) or set(order.iloc[:1])
    return c[c[ID].isin(keep)].reset_index(drop=True)


def run(ctx) -> dict:
    reg, v2r = begin(ctx, S)
    f = ctx.cfg.get_path("v2.foundation", {}) or {}
    jobs, elig = embedding_jobs(ctx)
    ctx.save_table(elig, S, "model_eligibility")
    if not jobs:
        for h in ("V2H09", "V2H10"):
            v2r.update_hypothesis(h, "unavailable", S, "no eligible (license + overlap verified) local foundation model")
        return {"status": "unavailable", "reason": "no eligible foundation model; see model_eligibility"}
    runner = EmbeddingV2Runner(ctx)
    perf = []
    for split in f.get("splits", ["train"]):
        out = runner.run(split)
        for k, v in out["models"].items():
            perf.append({"split": split, "model": k, **{kk: (str(vv) if isinstance(vv, (list, dict)) else vv) for kk, vv in v.items()}})
    perf = pd.DataFrame(perf)
    ctx.save_table(perf, S, "extraction_performance")
    if not perf.empty and "clips_per_s" in perf:
        d = perf.dropna(subset=["clips_per_s"])
        if not d.empty:
            save_figure_v2(ctx, S, "gpu_throughput", V.bar_ci(d, "model", "clips_per_s", None, None, "Embedding extraction throughput (clips/s)"),
                           d, "CPU/GPU computational performance")

    c = clip_frame(ctx)
    ctr = c[c["split"] == "train"].reset_index(drop=True)
    cs = _sample_students(ctr, int(f.get("layer_eval_clips", 6000)), ctx.seed)
    ev_l = LayerEvaluator(ctx, int(f.get("layer_eval_pca", 64)), int(f.get("knn_k", 5)), ctx.seed)
    rows, cka_rows = [], []
    best_per_model = {}
    for job in jobs:
        st = open_store(ctx, job)
        meta = st.meta()
        layers, pools = meta.get("layers", []), meta.get("poolings", [])
        mean_mats = {}
        for li, layer in enumerate(layers):
            for pj, pool in enumerate(pools):
                E = st.load(li, pj, ("train",))
                if E.empty:
                    continue
                j = cs.merge(E, on=FILE, how="inner").reset_index(drop=True)
                X = j[[x for x in E.columns if x != FILE]].to_numpy(np.float32)
                m = ev_l.evaluate(j, X)
                rows.append({"model": job["key"], "layer": int(layer), "layer_index": li, "pooling": pool, "pooling_index": pj,
                             "store_fp": st.fp, **m})
                if pool == "mean":
                    k = int(f.get("cka_clips", 2000))
                    mean_mats[int(layer)] = X[:k]
        if len(mean_mats) > 1:
            Ls = sorted(mean_mats)
            n = min(len(mean_mats[L]) for L in Ls)
            M = pd.DataFrame([[linear_cka(mean_mats[a][:n], mean_mats[b][:n]) for b in Ls] for a in Ls], index=Ls, columns=Ls)
            cka_rows.append(M.assign(model=job["key"]))
            save_figure_v2(ctx, S, f"layer_cka_{job['key']}", V.heatmap(M, f"Layer CKA similarity — {job['key']}", diverging=False, annotate=True),
                           M, "Representation similarity across layers (linear CKA, mean pooling)")
    lm = pd.DataFrame(rows)
    ctx.save_table(lm, S, "layer_metrics")
    if lm.empty:
        return {"status": "no embeddings available for evaluation"}
    for metric, title in (("correct_auc", "Correctness AUROC"), ("content_acc", "Item-identity kNN accuracy"), ("sr_auc", "Domain leakage AUROC")):
        d = lm.assign(series=lm["model"] + "/" + lm["pooling"])
        save_figure_v2(ctx, S, f"layer_relevance_{metric}", V.lines(d, "layer", metric, "series", f"{title} by layer"), d,
                       f"Layer-specific {title.lower()}")

    # ---- pre-registered layer choice for outcome evaluation
    sr_med = float(lm["sr_auc"].median())
    cand = lm[lm["sr_auc"] <= sr_med].sort_values("correct_auc", ascending=False).head(int(f.get("top_layers_for_outcome", 3)))
    if cand.empty:
        cand = lm.sort_values("correct_auc", ascending=False).head(1)
    ev = StudentLevelEvaluator(ctx, S)
    sch = schemes(ctx)
    base = {sc: ev.evaluate(ev.baseline_spec(), sc) for sc in sch}
    stu = []
    for r in cand.to_dict("records"):
        st = open_store(ctx, {"key": r["model"], "source": next(j["source"] for j in jobs if j["key"] == r["model"])})
        E = st.load(int(r["layer_index"]), int(r["pooling_index"]), ("train",))
        key = f"FM_{r['model']}_L{r['layer']}_{r['pooling']}"
        ev.add_provider(key, CachedProvider(FoundationProvider(ctx, key, E, int(f.get("student_pcs", 8)), 32,
                                                               store_fp=f"{st.fp}:{r['layer']}:{r['pooling']}")))
        rws, _ = evaluate_against(ev, CompactSpec(f"P20_{key}", "RIDGE", providers=[key],
                                                  budget=int(ctx.cfg.get_path("v2.evaluation.default_budget", 20)), fusion="residual"), base, sch)
        for x in rws:
            x.update({"model_key": r["model"], "layer": r["layer"], "pooling": r["pooling"], "layer_index": r["layer_index"],
                      "pooling_index": r["pooling_index"], "store_fp": st.fp, "correct_auc": r["correct_auc"], "sr_auc": r["sr_auc"]})
        stu += rws
    stu = pd.DataFrame(stu)
    ctx.save_table(stu, S, "student_results")
    prim = stu[stu["scheme"] == "repeated_kfold"].dropna(subset=["delta_r2"]) if not stu.empty and "delta_r2" in stu else pd.DataFrame()
    pick = (prim.sort_values(["delta_r2", "sr_auc"], ascending=[False, True]).iloc[0] if not prim.empty else cand.iloc[0])
    sel = {"model": str(pick.get("model_key", pick.get("model"))), "layer": int(pick["layer"]), "layer_index": int(pick["layer_index"]),
           "pooling": str(pick["pooling"]), "pooling_index": int(pick["pooling_index"]), "store_fp": str(pick["store_fp"]),
           "rule": "top correctness-AUROC layers among those with domain AUROC <= median, then max student ΔR² (ties: lower domain AUROC)"}
    ctx.save_json(sel, S, "selection")
    comp = lm[(lm["model"] == sel["model"]) & (lm["layer"] == sel["layer"])][["model", "layer", "pooling", "correct_auc", "pls_auc", "sr_auc", "content_acc"]]
    ctx.save_table(comp, S, "compression_comparison")
    # ---- embedding maps (selected representation)
    st = open_store(ctx, {"key": sel["model"], "source": next(j["source"] for j in jobs if j["key"] == sel["model"])})
    E = st.load(sel["layer_index"], sel["pooling_index"], ("train",))
    j = cs.merge(E, on=FILE, how="inner").head(3000)
    if len(j) > 10:
        from sklearn.decomposition import PCA

        X = j[[x for x in E.columns if x != FILE]].to_numpy(np.float32)
        xy = PCA(2, random_state=0).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-6))
        for lab_name, lab in (("task", j["task"].to_numpy()), ("correctness", np.where(j["item_score"] == 1, "correct", "incorrect")),
                              ("orig_rate", np.where(j["orig_sr_low"] == 1, "low_rate", "16k"))):
            save_figure_v2(ctx, S, f"embedding_map_{lab_name}", V.embedding_scatter(xy, lab, f"{sel['model']} L{sel['layer']} — by {lab_name}"),
                           pd.DataFrame({FILE: j[FILE], "pc1": xy[:, 0], "pc2": xy[:, 1], lab_name: lab}), "Embedding map (PCA)")
    if not prim.empty:
        b = prim.sort_values("delta_r2", ascending=False).iloc[0]
        v2r.update_hypothesis("V2H09", "supported" if b["delta_ci_low"] > 0 else "not_supported", S,
                              f"{b['name']} ΔR²={b['delta_r2']:.4f}", effect=float(b["delta_r2"]), ci_low=float(b["delta_ci_low"]),
                              ci_high=float(b["delta_ci_high"]))
    for mk, d in lm[lm["pooling"] == "mean"].groupby("model"):
        if d["layer"].nunique() > 2:
            bl = d.sort_values("correct_auc").iloc[-1]
            last = d[d["layer"] == d["layer"].max()].iloc[0]
            v2r.update_hypothesis("V2H10", "supported" if (bl["layer"] < last["layer"] and bl["correct_auc"] - last["correct_auc"] > 0.01) else "not_supported",
                                  S, f"{mk}: best layer {bl['layer']} AUROC {bl['correct_auc']:.3f} vs last {last['correct_auc']:.3f}",
                                  tier="exploratory", effect=float(bl["correct_auc"] - last["correct_auc"]))
            break
    return {"status": "ok", "selected": sel, "n_layer_evals": int(len(lm))}

"""Outcome-blind consensus diagnostics followed by clearly labeled outcome analyses.

Student pairs are dependent observations: never calculate naïve pair-level p-values
or treat 146,070 pairs as 146,070 independent training examples.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .m82_views import View


def attach_pair_diagnostics(pairs:pd.DataFrame,views:dict[str,View],ids,schools,
                            y=None,residual=None,instrument=None)->pd.DataFrame:
    p=pairs.copy();i=p.i.to_numpy(dtype=int);j=p.j.to_numpy(dtype=int)
    ids=np.asarray(ids).astype(str);schools=np.asarray(schools).astype(str)
    if len(ids)!=len(schools):raise ValueError('ids/schools length mismatch')
    if np.any(i==j):raise AssertionError('Self pair detected')
    if len(set(ids))!=len(ids):raise ValueError('Repeated student IDs')
    p['subject_a']=ids[i];p['subject_b']=ids[j]
    p['same_school']=schools[i]==schools[j]
    if instrument is not None:
        a=np.asarray(instrument).astype(str);p['same_instrument']=a[i]==a[j]
    for name,v in views.items():
        if v.matrix.shape[0]!=len(ids):raise ValueError('View length mismatch '+name)
        aa=v.matrix[i];bb=v.matrix[j]
        both=v.observed[i]&v.observed[j]
        n=np.sum(both,axis=1)
        delta=(aa-bb)**2*both
        p[f'distance__{name}']=np.sqrt(np.divide(delta.sum(axis=1),n,out=np.full(len(p),np.nan),where=n>0))
        p[f'joint_coverage__{name}']=n/max(1,len(v.columns))
    if y is not None:
        yy=np.asarray(y,float)
        if len(yy)!=len(ids):raise ValueError('Outcome size mismatch')
        p['score_gap']=np.abs(yy[i]-yy[j])
    if residual is not None:
        rr=np.asarray(residual,float)
        if len(rr)!=len(ids):raise ValueError('Residual size mismatch')
        p['screening_residual_gap']=np.abs(rr[i]-rr[j])
    return p


def summarize_pairs(p):
    rows=[]
    for kind,g in p.groupby('category',sort=False):
        row={'category':kind,'n_pairs':len(g),'n_students_involved':int(len(set(g.i)|set(g.j))),
             'pct_all_pairs':len(g)/len(p),'mean_consensus':g.consensus.mean(),
             'mean_chance_excess':g.chance_excess.mean(),
             'same_school_fraction':g.same_school.mean() if 'same_school' in g else np.nan,
             'mean_screening_distance':g.distance__screening.mean() if 'distance__screening' in g else np.nan}
        for col in ('score_gap','screening_residual_gap'):
            row['median_'+col]=g[col].median() if col in g else np.nan
            row['mean_'+col]=g[col].mean() if col in g else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values('n_pairs',ascending=False)


def comparison_descriptive(p,seed=2026,n_permutations=100):
    """Exploratory: student-indexed outcome permutations within screening quantiles.

    Permutation is conditional on coarse screening similarity only; it is not a
    causal test, and it does not jointly condition on schools and instruments.
    """
    if 'screening_residual_gap' not in p or 'distance__screening' not in p:
        return pd.DataFrame()
    screening=p.distance__screening.to_numpy(float)
    finite=np.isfinite(screening)
    if not finite.any():return pd.DataFrame()
    cutoff=np.quantile(screening[finite],.25)
    near=finite & (screening<=cutoff)
    r=[]
    for kind in ('stable_together','stable_apart','cross_view_disagreement','uncertain'):
        sel=near & (p.category.to_numpy()==kind)
        if not sel.any():continue
        z=p.loc[sel]
        r.append({'category':kind,'n_pairs_matched':len(z),
                  'median_residual_gap_matched':float(z.screening_residual_gap.median()),
                  'mean_residual_gap_matched':float(z.screening_residual_gap.mean()),
                  'mean_outcome_gap_matched':float(z.score_gap.mean()) if 'score_gap' in z else np.nan,
                  'same_school_fraction_matched':float(z.same_school.mean())})
    return pd.DataFrame(r)


def stable_graph_nodes(p,n_students:int):
    """Stable-pair graph components; singletons stay unassigned, no forced k."""
    edges=p[p.category=='stable_together']
    a=edges.i.to_numpy(int);b=edges.j.to_numpy(int)
    if n_students==0:return pd.DataFrame()
    u=np.r_[a,b];v=np.r_[b,a]
    graph=csr_matrix((np.ones(len(u),dtype=np.uint8),(u,v)),shape=(n_students,n_students))
    n,labels=connected_components(graph,directed=False)
    size=np.bincount(labels,minlength=n)
    return pd.DataFrame({'i':np.arange(n_students),'component':labels,
                         'component_size':size[labels],
                         'graph_assigned':size[labels]>=2})


def consensus_score_relationship(p):
    """Dependence-aware wording: correlations are descriptive only."""
    cols=['score_gap','screening_residual_gap']
    rows=[]
    good=p.consensus.notna()
    for target in cols:
        if target not in p:continue
        a=p.loc[good,'consensus'];b=p.loc[good,target]
        rho=float(spearmanr(a,b).statistic) if len(a)>10 and a.nunique()>1 else np.nan
        rows.append({'target':target,'descriptive_pairwise_spearman':rho,
                     'pair_count_not_independent':int(len(a)),
                     'interpretation':'Exploratory only: pairs share students; do not report pair-level p-values'})
    return pd.DataFrame(rows)


def school_confound_summary(p):
    rows=[]
    for flag,g in p.groupby('same_school'):
        rows.append({'same_school':bool(flag),'n_pairs':len(g),
                     'mean_consensus':float(g.consensus.mean()),
                     'stable_together_fraction':float((g.category=='stable_together').mean()),
                     'stable_apart_fraction':float((g.category=='stable_apart').mean())})
    return pd.DataFrame(rows)
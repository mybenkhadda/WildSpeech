"""WildSpeech M8.3: leakage-conscious performance bottleneck diagnosis.

Reads the existing M8 student-level features (no audio extraction). Writes only
aggregate diagnostics and anonymous OOF identifiers. No test outcomes required.
Run via python -m wildspeech_m8.m83_bottleneck --help after adding to package.
"""
from __future__ import annotations
import argparse
import json
import re
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

try:
    from .m8_pipeline import (Config, prepare_dataset, FoldEncoder,
                              make_estimator, choose_features, feature_families,
                              assert_safe_features)
except ImportError:
    from wildspeech_m8.m8_pipeline import (Config, prepare_dataset, FoldEncoder,
                                           make_estimator, choose_features,
                                           feature_families, assert_safe_features)

@dataclass
class AuditConfig:
    data_root: str = '/content/drive/MyDrive/WildSpeech'
    output_root: str = '/content/drive/MyDrive/WildSpeech_M8'
    profile_csv: str = ''
    seed: int = 2026
    folds: int = 5
    repeats: int = 2
    max_features: int = 450
    quick: bool = False
    bootstrap: int = 400
    pair_sample: int = 150000
    run_residual_probe: bool = True
    run_learning_curve: bool = True
    run_negative_control: bool = True


def safe_columns(frame: pd.DataFrame, limit: int) -> tuple[list[str],dict]:
    """Reuse M8 exclusion logic; explicit empty-family reports."""
    safe = choose_features(frame, limit)
    assert_safe_features(safe)
    safe = [c for c in safe if c in frame.columns]
    if not safe: raise ValueError('No usable feature columns')
    families = feature_families(safe)
    return safe, families


def make_splits(n: int, groups, cfg: AuditConfig, grouped: bool):
    if n < 8: raise ValueError('At least eight rows needed')
    rng = np.random.default_rng(cfg.seed)
    if grouped:
        distinct = np.unique(groups)
        if len(distinct)<3: return []
        folds = min(cfg.folds,len(distinct))
        out=[]
        for rep in range(cfg.repeats):
            # GroupKFold uses group ID ordering; remap in a seeded random order.
            order = rng.permutation(distinct)
            remap={g:i for i,g in enumerate(order)}
            pseudo=np.array([remap[g] for g in groups])
            for fold,(tr,va) in enumerate(GroupKFold(n_splits=folds).split(np.arange(n),groups=pseudo)):
                if set(groups[tr])&set(groups[va]): raise AssertionError('Group overlap')
                out.append((rep,fold,tr,va))
        return out
    out=[]
    for rep in range(cfg.repeats):
        cv=KFold(n_splits=min(cfg.folds,n//3),shuffle=True,random_state=cfg.seed+rep)
        for fold,(tr,va) in enumerate(cv.split(np.arange(n))):
            out.append((rep,fold,tr,va))
    return out


def fit_predict(frame, y, train, valid, columns, seed, baseline=False):
    if baseline or not columns:
        p=np.full(len(valid),float(np.mean(y[train])))
        return p,np.full(len(train),float(np.mean(y[train])))
    enc=FoldEncoder().fit(frame.iloc[train][columns])
    xtr=enc.transform(frame.iloc[train][columns]); xva=enc.transform(frame.iloc[valid][columns])
    model=make_estimator(seed,fast=True)
    model.fit(xtr,y[train])
    return np.asarray(model.predict(xva),float),np.asarray(model.predict(xtr),float)


def metric(y,p):
    return {'mae':float(mean_absolute_error(y,p)),
            'rmse':float(np.sqrt(mean_squared_error(y,p))),
            'r2':float(r2_score(y,p)) if len(y)>1 and np.std(y)>1e-12 else np.nan}


def build_sets(columns,family):
    fams=sorted(set(family.values()))
    sets={'all':columns,'screening_only':[c for c in columns if family[c]=='screening']}
    if not sets['screening_only']: sets.pop('screening_only')
    for fam in fams:
        subset=[c for c in columns if family[c]==fam]
        if fam!='screening' and subset:sets[f'only_{fam}']=subset
        removal=[c for c in columns if family[c]!=fam]
        if removal and fam!='screening':sets[f'without_{fam}']=removal
    return sets


def cv_tournament(frame,y,groups,columns,family,cfg,grouped):
    sets=build_sets(columns,family)
    splits=make_splits(len(y),groups,cfg,grouped)
    if not splits:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    preds=[];folds=[];gaps=[]
    for rep,fold,tr,va in tqdm(splits,desc=f"M8.3 {'school-disjoint' if grouped else 'student'} CV",unit="fold"):
        split_label='grouped' if grouped else 'random'
        for name,cols in [('mean',[]),*sets.items()]:
            p,pt=fit_predict(frame,y,tr,va,cols,cfg.seed+1000*rep+fold,baseline=(name=='mean'))
            m=metric(y[va],p); mt=metric(y[tr],pt)
            folds.append({'split':split_label,'repeat':rep,'fold':fold,'model':name,
                          'n_train':len(tr),'n_valid':len(va),'train_mae':mt['mae'],
                          'train_rmse':mt['rmse'],**m})
            gaps.append({'split':split_label,'repeat':rep,'fold':fold,'model':name,
                         'generalization_gap_mae':m['mae']-mt['mae']})
            for ix,yp in zip(va,p):
                preds.append({'split':split_label,'repeat':rep,'fold':fold,'row':int(ix),
                              'model':name,'target':float(y[ix]),'pred':float(yp),
                              'group':str(groups[ix])})
    return pd.DataFrame(preds),pd.DataFrame(folds),pd.DataFrame(gaps)


def cluster_bootstrap(oof:pd.DataFrame, reference='screening_only', challenger='all', n=400, seed=2026):
    """One prediction per student for each repeat; paired row/group resampling.
    Distinct repeated OOF predictions are never treated as independent labels.
    """
    sub=oof[oof.model.isin([reference,challenger])]
    if not {reference,challenger}.issubset(set(sub.model)):return pd.DataFrame()
    pivot=sub.pivot_table(index=['repeat','row','group'],columns='model',values=['pred','target'],aggfunc='first')
    if ('pred',reference) not in pivot or ('pred',challenger) not in pivot:return pd.DataFrame()
    v=pivot.reset_index(); rows=[]; rng=np.random.default_rng(seed)
    for rep,d in v.groupby('repeat'):
        y=d[('target',reference)].to_numpy(float)
        p0=d[('pred',reference)].to_numpy(float)
        p1=d[('pred',challenger)].to_numpy(float)
        group=d[('group','')].astype(str).to_numpy()
        for unit in ('student','group'):
            unique=np.unique(group); use_groups=(unit=='group' and len(unique)>=3)
            for boot in range(n):
                if use_groups:
                    picked=rng.choice(unique,len(unique),replace=True)
                    ii=np.concatenate([np.where(group==g)[0] for g in picked])
                else:ii=rng.integers(0,len(y),len(y))
                rows.append({'repeat':int(rep),'unit':unit,'boot':boot,
                             'mae_improvement':float(np.mean(np.abs(y[ii]-p0[ii]))-
                                                     np.mean(np.abs(y[ii]-p1[ii])))})
    return pd.DataFrame(rows)


def pair_diagnostics(oof, cfg):
    """Descriptive pair assessment; independent units are STUDENTS, not pairs."""
    d=oof[(oof.model=='all')&(oof.repeat==0)].sort_values('row')
    if len(d)<3:return pd.DataFrame(),pd.DataFrame()
    y=d.target.to_numpy(); p=d.pred.to_numpy(); n=len(y)
    rng=np.random.default_rng(cfg.seed+11)
    total=n*(n-1)//2; take=min(cfg.pair_sample,total)
    if total<=cfg.pair_sample: i,j=np.triu_indices(n,1)
    else:
        # Draw without storing the n*n matrix; duplicates may occur and are deduplicated.
        i=rng.integers(0,n,2*take); j=rng.integers(0,n,2*take)
        mask=i!=j; i,j=np.minimum(i[mask],j[mask]),np.maximum(i[mask],j[mask])
        pairs=np.unique(np.column_stack([i,j]),axis=0)[:take];i,j=pairs[:,0],pairs[:,1]
    gap=np.abs(y[i]-y[j]); dp=p[i]-p[j]; dy=y[i]-y[j]
    scored=np.abs(dy)>1e-10
    ranks=np.where(scored, (np.sign(dp)==np.sign(dy)).astype(float),np.nan)
    bins=pd.qcut(pd.Series(gap),q=min(5,len(np.unique(gap))),duplicates='drop')
    x=pd.DataFrame({'gap':gap,'correct':ranks,'gap_bin':bins.astype(str)})
    result=x.groupby('gap_bin',observed=True).agg(n_pairs=('gap','size'),
              median_true_gap=('gap','median'),rank_accuracy=('correct','mean')).reset_index()
    # Explicit non-inferential descriptive output only.
    overall=pd.DataFrame([{'n_students':n,'n_pairs':len(x),'rank_accuracy':float(np.nanmean(ranks)),
                           'note':'Pairs share students; do not compute ordinary pair-level confidence intervals.'}])
    return result,overall


def subgroup_analysis(oof, frame, family, cfg):
    d=oof[(oof.model=='all')&(oof.repeat==0)].copy()
    d['error']=d.target-d.pred; d['abs_error']=d.error.abs()
    outs=[]
    for name,vals in [('score',d.target),('prediction',d.pred)]:
        b=pd.qcut(pd.Series(vals.to_numpy()),q=4,duplicates='drop').astype(str).to_numpy()
        for key in np.unique(b):
            x=d[b==key]
            outs.append({'dimension':name,'segment':key,'n':len(x),
                         'mae':float(x.abs_error.mean()),'bias':float(x.error.mean())})
    for key,g in d.groupby('group'):
        if len(g)>=3:outs.append({'dimension':'school','segment':str(key),
                                 'n':len(g),'mae':float(g.abs_error.mean()),
                                 'bias':float(g.error.mean())})
    # Feature quantiles evaluated only after OOF, descriptive not a validation metric.
    feature_names=[c for c in frame.columns if family.get(c) in ('audio','speech_text','temporal')]
    for col in feature_names[:12]:
        x=pd.to_numeric(frame.iloc[d.row.to_numpy()][col],errors='coerce').to_numpy()
        if np.isfinite(x).sum()<max(20,len(x)//2):continue
        q=pd.qcut(pd.Series(x),q=3,duplicates='drop').astype(str).to_numpy()
        for seg in np.unique(q):
            z=d[q==seg]
            if len(z)>=5:outs.append({'dimension':col,'segment':seg,
                                     'n':len(z),'mae':float(z.abs_error.mean()),
                                     'bias':float(z.error.mean())})
    return pd.DataFrame(outs)


def residual_probe(frame,y,groups,columns,cfg,grouped=False):
    """For each outer fold: crossfit *training* residuals, fit residual model,
    evaluate on untouched outer validation. No outer labels used in training.
    """
    splits=make_splits(len(y),groups,cfg,grouped)
    if not splits:return pd.DataFrame()
    rows=[]
    for rep,fold,tr,va in splits[:min(len(splits),5 if cfg.quick else 10)]:
        p_outer,_=fit_predict(frame,y,tr,va,columns,cfg.seed+fold)
        inner=GroupKFold(n_splits=min(3,len(np.unique(groups[tr])))) if grouped else KFold(n_splits=3,shuffle=True,random_state=cfg.seed+fold)
        resid=np.full(len(tr),np.nan)
        itr=inner.split(tr,groups=groups[tr]) if grouped else inner.split(tr)
        for a,b in itr:
            pa,_=fit_predict(frame,y,tr[a],tr[b],columns,cfg.seed+100+fold)
            resid[b]=y[tr[b]]-pa
        if not np.isfinite(resid).all():raise AssertionError('Nested OOF missing')
        enc=FoldEncoder().fit(frame.iloc[tr][columns]);xtr=enc.transform(frame.iloc[tr][columns]);xva=enc.transform(frame.iloc[va][columns])
        probe=Ridge(alpha=100.0).fit(xtr,resid)
        adjusted=p_outer+np.clip(probe.predict(xva),-2*np.std(resid),2*np.std(resid))
        a=metric(y[va],p_outer);b=metric(y[va],adjusted)
        rows.append({'split':'grouped' if grouped else 'random','repeat':rep,'fold':fold,
                     'base_mae':a['mae'],'residual_corrected_mae':b['mae'],
                     'mae_improvement':a['mae']-b['mae'],
                     'base_rmse':a['rmse'],'residual_corrected_rmse':b['rmse']})
    return pd.DataFrame(rows)


def negative_control(frame,y,groups,columns,cfg):
    """Shuffle target within training only; compare genuine vs null holdout accuracy."""
    rows=[];rng=np.random.default_rng(cfg.seed+125)
    for _,fold,tr,va in make_splits(len(y),groups,cfg,False)[:min(3,cfg.folds)]:
        pred,_=fit_predict(frame,y,tr,va,columns,cfg.seed+fold)
        shuffled=y.copy();shuffled[tr]=rng.permutation(y[tr])
        null,_=fit_predict(frame,shuffled,tr,va,columns,cfg.seed+fold)
        rows.append({'fold':fold,'true_mae':metric(y[va],pred)['mae'],
                     'null_mae':metric(y[va],null)['mae']})
    return pd.DataFrame(rows)


def learning_curve(frame,y,groups,columns,cfg):
    rows=[];rng=np.random.default_rng(cfg.seed+400)
    splits=make_splits(len(y),groups,cfg,False)[:min(3,cfg.folds)]
    for _,fold,tr,va in splits:
        for ratio in (.25,.5,.75,1.):
            take=rng.choice(tr,max(8,int(len(tr)*ratio)),replace=False)
            p,t=fit_predict(frame,y,take,va,columns,cfg.seed+fold)
            rows.append({'fold':fold,'train_fraction':ratio,'train_n':len(take),
                         'valid_mae':metric(y[va],p)['mae'],
                         'train_mae':metric(y[take],t)['mae']})
    return pd.DataFrame(rows)


def residual_clusters(oof,frame,columns,cfg):
    """Unsupervised geometry fitted on training features only per outer fold;
    labels overlayed with OOF errors later. Descriptive, not score-optimized.
    """
    d=oof[(oof.model=='all')&(oof.repeat==0)].copy(); results=[]
    for fold, val in d.groupby('fold'):
        va=val.row.to_numpy(int);tr=np.setdiff1d(np.arange(len(frame)),va)
        if len(tr)<15 or len(va)<4:continue
        enc=FoldEncoder().fit(frame.iloc[tr][columns]); ztr=enc.transform(frame.iloc[tr][columns]);zva=enc.transform(frame.iloc[va][columns])
        pca=PCA(n_components=min(8,ztr.shape[1],len(tr)-1),random_state=cfg.seed).fit(ztr)
        u=pca.transform(ztr);v=pca.transform(zva)
        km=KMeans(n_clusters=min(4,max(2,len(tr)//15)),random_state=cfg.seed+n_fold(fold),n_init=10).fit(u)
        labels=km.predict(v)
        for k in np.unique(labels):
            x=val.iloc[np.where(labels==k)[0]]
            results.append({'fold':int(fold),'cluster':int(k),'n':len(x),
                            'mae':float(np.abs(x.target-x.pred).mean()),
                            'bias':float((x.target-x.pred).mean())})
    return pd.DataFrame(results)

def n_fold(x):return int(x)


def write_figures(out,folds,subgroups,pair,curve,probe):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
    figs=[]
    def save(name):
        plt.tight_layout();plt.savefig(figdir/name,dpi=145,bbox_inches='tight');plt.close();figs.append(name)
    if not folds.empty:
        agg=folds[folds.split=='random'].groupby('model').mae.agg(['mean','std']).sort_values('mean').head(16)
        plt.figure(figsize=(11,5));plt.barh(agg.index,agg['mean'],xerr=agg['std'].fillna(0));plt.gca().invert_yaxis();plt.xlabel('Validation MAE (lower better)');plt.title('Feature family ablations');save('01_feature_ablation.png')
        x=folds[folds.model=='all'].groupby('split')[['mae','train_mae']].mean()
        x.plot(kind='bar',figsize=(7,4),rot=0);plt.ylabel('MAE');plt.title('Generalization gap by split');save('02_generalization_gap.png')
    if not subgroups.empty:
        s=subgroups[subgroups.dimension=='score'];plt.figure(figsize=(9,4));plt.bar(s.segment,s.mae);plt.xticks(rotation=20,ha='right');plt.ylabel('MAE');plt.title('Error by true-score quartile');save('03_score_regimes.png')
        s=subgroups[subgroups.dimension=='school'].sort_values('mae',ascending=False).head(20)
        if not s.empty:
            plt.figure(figsize=(10,4));plt.bar(range(len(s)),s.mae);plt.xticks(range(len(s)),['group '+str(i+1) for i in range(len(s))],rotation=70);plt.ylabel('MAE');plt.title('School heterogeneity (anonymized)');save('04_school_heterogeneity.png')
    if not pair.empty:
        plt.figure(figsize=(9,4));plt.plot(pair.median_true_gap,pair.rank_accuracy,'o-');plt.ylim(0,1);plt.ylabel('Pair ranking accuracy');plt.xlabel('Median true-score gap');plt.title('Ranking difficulty');save('05_pair_difficulty.png')
    if not curve.empty:
        x=curve.groupby('train_fraction')[['train_mae','valid_mae']].mean();x.plot(marker='o',figsize=(7,4));plt.ylabel('MAE');plt.xlabel('Training sample fraction');plt.title('Learning curve');save('06_learning_curve.png')
    if not probe.empty:
        x=probe[['base_mae','residual_corrected_mae']].mean();plt.figure(figsize=(6,4));plt.bar(x.index,x.values);plt.xticks(rotation=15);plt.ylabel('MAE');plt.title('Nested residual predictability');save('07_residual_probe.png')
    return figs


def evidence_summary(folds,boots,probe,negative,curve,grouped_available):
    notes=[]
    if grouped_available:
        r=folds[(folds.split=='random')&(folds.model=='all')].mae.mean()
        g=folds[(folds.split=='grouped')&(folds.model=='all')].mae.mean()
        notes.append(f'Random-CV MAE {r:.4f}; held-out-school CV MAE {g:.4f}. These are different validation estimands.')
    else: notes.append('Held-out-school validation unavailable: fewer than 3 distinct schools.')
    if not boots.empty:
        b=boots[(boots.unit=='group')&(boots.repeat==0)]
        if b.empty:b=boots[(boots.unit=='student')&(boots.repeat==0)]
        lo,med,hi=np.percentile(b.mae_improvement,[2.5,50,97.5]);notes.append(f'All features vs screening baseline: bootstrap MAE gain median {med:.4f}, 95% percentile interval [{lo:.4f}, {hi:.4f}].')
        if lo<=0<=hi:notes.append('Incremental feature benefit is uncertain under the chosen resampling unit.')
    if not probe.empty:
        notes.append(f'Nested residual correction: mean MAE gain {probe.mae_improvement.mean():+.4f}; positive is better. A negative gain implies no actionable residual signal under this probe.')
    if not negative.empty and negative.true_mae.mean()>=negative.null_mae.mean():
        notes.append('WARNING: Genuine model failed to beat shuffled-training-target negative control. Inspect data joins, features and validation.')
    if not curve.empty:
        slope=curve.groupby('train_fraction').valid_mae.mean()
        if len(slope)>=2 and slope.iloc[-1]<slope.iloc[0]:notes.append('Learning-curve MAE improves with more training data; a data-size bottleneck remains plausible.')
    return notes


def run_frame(frame,y,groups,out,cfg:AuditConfig):
    out=Path(out);(out/'tables').mkdir(parents=True,exist_ok=True);(out/'reports').mkdir(parents=True,exist_ok=True)
    frame=frame.reset_index(drop=True).copy();y=np.asarray(y,float);groups=np.asarray(groups).astype(str)
    if len(frame)!=len(y) or len(groups)!=len(y) or not np.isfinite(y).all():raise ValueError('Alignment/nonfinite outcome error')
    if len(set(groups))<2:warnings.warn('Only one school, school shift cannot be measured')
    cols,fam=safe_columns(frame,cfg.max_features)
    print(f"[M8.3] {len(frame)} students | {len(cols)} features | {len(set(groups))} schools",flush=True)
    print("[M8.3] Phase 1/8: random-student ablations",flush=True)
    rand_p,rand_f,gap=cv_tournament(frame,y,groups,cols,fam,cfg,False)
    grouped=len(np.unique(groups))>=3
    print("[M8.3] Phase 2/8: school-held-out ablations",flush=True)
    group_p,group_f,_=cv_tournament(frame,y,groups,cols,fam,cfg,True) if grouped else (pd.DataFrame(),pd.DataFrame(),pd.DataFrame())
    oof=pd.concat([rand_p,group_p],ignore_index=True)
    folds=pd.concat([rand_f,group_f],ignore_index=True)
    base='screening_only' if 'screening_only' in set(rand_p.model) else 'mean'
    print("[M8.3] Phase 3/8: paired bootstrap + pairwise ranking",flush=True)
    boots=cluster_bootstrap(rand_p,base,'all',cfg.bootstrap,cfg.seed)
    pairs,pair_overall=pair_diagnostics(rand_p,cfg)
    subgroup=subgroup_analysis(rand_p,frame,fam,cfg)
    print("[M8.3] Phase 4/8: nested residual probes",flush=True)
    probe=residual_probe(frame,y,groups,cols,cfg,False) if cfg.run_residual_probe else pd.DataFrame()
    if grouped and cfg.run_residual_probe:
        probe=pd.concat([probe,residual_probe(frame,y,groups,cols,cfg,True)],ignore_index=True)
    print("[M8.3] Phase 5/8: target permutation control",flush=True)
    null=negative_control(frame,y,groups,cols,cfg) if cfg.run_negative_control else pd.DataFrame()
    print("[M8.3] Phase 6/8: data-size learning curves",flush=True)
    curve=learning_curve(frame,y,groups,cols,cfg) if cfg.run_learning_curve else pd.DataFrame()
    print("[M8.3] Phase 7/8: cluster residual overlays",flush=True)
    clusters=residual_clusters(rand_p,frame,cols,cfg)
    tables={'fold_metrics':folds,'oof_predictions':oof,'paired_bootstrap':boots,
            'pair_gap_diagnostics':pairs,'pair_overall':pair_overall,'subgroup_errors':subgroup,
            'residual_probe':probe,'negative_control':null,'learning_curve':curve,
            'feature_clusters':clusters,'generalization_gaps':gap}
    for name,d in tables.items():
        if not d.empty:
            if name=='oof_predictions':
                # Deidentify local subjects while preserving repeat/row join semantics;
                # group can itself identify school, so never publish this file.
                d.to_csv(out/'tables'/f'{name}_PRIVATE.csv',index=False)
            else:d.to_csv(out/'tables'/f'{name}.csv',index=False)
    print("[M8.3] Phase 8/8: diagnostics + figures",flush=True)
    figures=write_figures(out,folds,subgroup,pairs,curve,probe)
    manifest={'rows':len(frame),'schools':int(len(np.unique(groups))),
              'feature_count':len(cols),'families':pd.Series(list(fam.values())).value_counts().to_dict(),
              'feature_names':cols,'config':asdict(cfg),'figures':figures,
              'notes':evidence_summary(folds,boots,probe,null,curve,grouped)}
    (out/'reports'/'diagnostic_manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    lines=['# WildSpeech M8.3 — Performance Bottleneck Audit','',f'Rows: {len(frame)}, schools: {len(np.unique(groups))}, features: {len(cols)}.','',
           '## Key diagnostic evidence','',*['- '+x for x in manifest['notes']], '',
           '## Main model averages','',folds.groupby(['split','model'])[['mae','rmse','r2','train_mae']].mean().round(4).to_string(),'',
           '## Interpretation','',
           '- Held-out-school splits and random splits answer different questions; compare separately.',
           '- All feature selection is label-independent; feature preprocessing is refit inside each fold.',
           '- Pairwise metrics are descriptive because pairs reuse students.',
           '- The residual probe uses inner training-only OOF residuals, then tests on outer validation.',
           '- Clusters are outcome-blind, fit on outer training features, and interpreted only after prediction.',
           '- No genuine competition-data scores are claimed from synthetic tests.',
           '- OOF predictions include sensitive group structure; retain privately and do not commit.',
           '- Bootstraps measure uncertainty conditional on the chosen splits, not a new external test set.']
    (out/'M83_BOTTLENECK_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    return manifest


def run_official(cfg:AuditConfig):
    base=Config(data_root=cfg.data_root,output_root=cfg.output_root,
                profile_csv=cfg.profile_csv,quick=cfg.quick,max_features=cfg.max_features)
    frame,test,y,schools,fmt,clips,manifest=prepare_dataset(base)
    return run_frame(frame,y,schools,Path(cfg.output_root)/'m83_bottleneck',cfg)


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root');parser.add_argument('--output-root');parser.add_argument('--profile-csv')
    parser.add_argument('--seed',type=int);parser.add_argument('--folds',type=int)
    parser.add_argument('--repeats',type=int);parser.add_argument('--max-features',type=int)
    parser.add_argument('--bootstrap',type=int);parser.add_argument('--quick',action='store_true')
    parser.add_argument('--no-residual-probe',action='store_true');parser.add_argument('--no-learning-curve',action='store_true')
    args=parser.parse_args(); options={k:v for k,v in vars(args).items() if v is not None and k not in ('no_residual_probe','no_learning_curve')}
    options['run_residual_probe']=not args.no_residual_probe;options['run_learning_curve']=not args.no_learning_curve
    print(json.dumps(run_official(AuditConfig(**options)),indent=2,default=str))

if __name__=='__main__':cli()
"""M8.1 research atlas: paired feature ablations, conditional associations,
negative controls, out-of-school robustness, stability, and diagnostic figures.

Independent analysis of the M8 pipeline; no target-informed feature selection
outside cross-validation. Descriptive subgroup fits are explicitly tagged.
"""
from __future__ import annotations
import argparse
import json
import math
import warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, adjusted_rand_score, normalized_mutual_info_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .m8_pipeline import (Config, prepare_dataset, feature_families, FoldEncoder,
                         make_estimator, score, crossfit_baseline,
                         ResidualDiscovery, safe_save, report_json)

VERSION='M8.1_exp1'

@dataclass
class ResearchConfig:
    folds: int=5
    repeats: int=2
    seed: int=2027
    permutations: int=3
    bootstrap: int=800
    stability_resamples: int=25
    max_assoc_features: int=350
    quick: bool=False


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values, preserving nan positions."""
    arr=np.asarray(p,dtype=float);valid=np.isfinite(arr);out=np.full(arr.shape,np.nan,dtype=float)
    if not valid.any():return out
    vals=arr[valid];idx=np.argsort(vals);s=vals[idx];n=len(s)
    adjusted=np.minimum.accumulate((s*n/np.arange(1,n+1))[::-1])[::-1]
    corrected=np.empty(n);corrected[idx]=np.minimum(1,adjusted);out[valid]=corrected
    return out


def _family_cols(df):
    fam=feature_families(df.columns)
    return {f:[c for c in df if fam[c]==f] for f in sorted(set(fam.values()))}


def feature_sets(df):
    """Predeclared sets; no label-based screening of individual features."""
    fam=_family_cols(df)
    screen=fam.get('screening',[]);context=fam.get('context_behavior',[])
    temp=fam.get('temporal',[]);audio=fam.get('audio',[])
    speech=fam.get('speech_text',[])
    strict=screen or (screen+context) or list(df.columns)
    base=list(dict.fromkeys(screen+context)) if screen else list(df.columns)
    result={'screening_only':strict,'screening_context':base,'all_features':list(df.columns)}
    for name,cols in [('temporal',temp),('acoustic',audio),('speech_text',speech)]:
        if cols:result['base_plus_'+name]=list(dict.fromkeys(base+cols))
        if cols and len(base)<df.shape[1]:result['all_minus_'+name]=[c for c in df.columns if c not in cols]
    if audio and speech:result['base_plus_audio_speech']=list(dict.fromkeys(base+audio+speech))
    # Silently do not invent modality data; deduplicate equal sets by retaining named
    # aliases for the report while fitting each distinct feature set just once.
    return {k:v for k,v in result.items() if v}


def _model_prediction(train,val,ytr,seed,fast):
    enc=FoldEncoder().fit(train);xt=enc.transform(train);xv=enc.transform(val)
    est=make_estimator(seed,fast).fit(xt,ytr)
    return np.asarray(est.predict(xv),dtype=float)


def experiment_ablation(df,y,schools,output,cfg:ResearchConfig):
    """Same student folds, identical estimator, feature-family ablation + controls.

    Permuted acoustic/speech features are shuffled independently within train
    and validation partitions, never using validation labels or their values to
    fit training preprocessing.
    """
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    sets=feature_sets(df)
    if cfg.quick:
        cv=KFold(n_splits=min(cfg.folds,3),shuffle=True,random_state=cfg.seed)
        splits=[(0,j,a,b) for j,(a,b) in enumerate(cv.split(df))]
    else:
        cv=RepeatedKFold(n_splits=cfg.folds,n_repeats=cfg.repeats,random_state=cfg.seed)
        splits=[(j//cfg.folds,j%cfg.folds,a,b) for j,(a,b) in enumerate(cv.split(df))]
    cache={tuple(v):None for v in sets.values()}
    alias={name:tuple(cols) for name,cols in sets.items()}
    unique=list(cache)
    rng=np.random.default_rng(cfg.seed)
    predrows=[];foldrows=[]
    for repeat,fold,tr,va in splits:
        print(f'[M8.1 ablation] repeat={repeat} fold={fold} train={len(tr)} val={len(va)}',flush=True)
        yt=y[va]
        local={}
        for j,cols in enumerate(unique):
            pred=_model_prediction(df.iloc[tr][list(cols)],df.iloc[va][list(cols)],y[tr],cfg.seed+repeat*100+fold*11+j,cfg.quick)
            local[cols]=pred
        for name,cols in alias.items():
            p=local[cols]
            foldrows.append({'repeat':repeat,'fold':fold,'experiment':name,'n_features':len(cols),**score(yt,p)})
            predrows += [{'subject_id':str(df.index[ix]),'school':str(schools[ix]),'repeat':repeat,'fold':fold,
                          'experiment':name,'y':float(y[ix]),'pred':float(v)} for ix,v in zip(va,p)]
        controlled=[name for name in ['audio','speech_text'] if _family_cols(df).get(name)]
        for family in controlled:
            columns=_family_cols(df)[family]
            for perm in range(1 if cfg.quick else cfg.permutations):
                dtr=df.iloc[tr].copy();dva=df.iloc[va].copy()
                perm_tr=rng.permutation(len(dtr));perm_va=rng.permutation(len(dva))
                dtr.loc[:,columns]=dtr[columns].to_numpy()[perm_tr]
                dva.loc[:,columns]=dva[columns].to_numpy()[perm_va]
                p=_model_prediction(dtr,dva,y[tr],cfg.seed+repeat*101+fold*13+perm+1300,cfg.quick)
                name=f'control_shuffle_{family}_{perm}'
                foldrows.append({'repeat':repeat,'fold':fold,'experiment':name,'n_features':df.shape[1],**score(yt,p)})
                predrows += [{'subject_id':str(df.index[ix]),'school':str(schools[ix]),'repeat':repeat,'fold':fold,
                              'experiment':name,'y':float(y[ix]),'pred':float(v)} for ix,v in zip(va,p)]
    pr=pd.DataFrame(predrows);ft=pd.DataFrame(foldrows)
    safe_save(pr,out/'ablation_oof.csv');safe_save(ft,out/'ablation_fold_metrics.csv')
    comparison=[]
    pivot=ft.pivot(index=['repeat','fold'],columns='experiment',values='r2')
    for name in pivot.columns:
        if name=='screening_context':baseline='screening_only'
        else:baseline='screening_context' if 'screening_context' in pivot else 'all_features'
        s=(pivot[name]-pivot[baseline]).dropna() if name!=baseline else pd.Series(dtype=float)
        comparison.append({'experiment':name,'reference':baseline,'mean_r2':pivot[name].mean(),
                          'sd_r2':pivot[name].std(ddof=1),
                          'mean_paired_delta_r2':s.mean() if len(s) else np.nan,
                          'paired_delta_sd':s.std(ddof=1) if len(s)>1 else np.nan,
                          'folds_better':int((s>0).sum()) if len(s) else 0,
                          'folds_compared':int(len(s)),'features':int(ft[ft.experiment==name].n_features.iloc[0])})
    sm=pd.DataFrame(comparison).sort_values('mean_r2',ascending=False)
    safe_save(sm,out/'ablation_summary.csv')
    report_json({'sets':{k:len(v) for k,v in sets.items()},'controls':cfg.permutations,
                 'n_splits':len(splits),'classification':'predeclared feature-family comparisons; no outer-fold choice for submission'},out/'ablation_manifest.json')
    return pr,ft,sm


def _pick_screening(df):
    sets=feature_sets(df)
    return sets.get('screening_context') or sets['screening_only']


def conditional_associations(df,y,output,cfg:ResearchConfig):
    """Double cross-fitting, y residualized via screening CatBoost, features via
    screening Ridge. Conditional associations are exploratory, not causal.
    """
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    baseline=_pick_screening(df)
    xcols=[c for c in df.select_dtypes(include='number').columns if c not in baseline]
    if len(xcols)>cfg.max_assoc_features:xcols=xcols[:cfg.max_assoc_features]
    if not xcols:return pd.DataFrame(),pd.DataFrame()
    xmat=df[xcols].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)
    oof_y=np.full(len(y),np.nan);oof_fx=np.full_like(xmat,np.nan)
    folds=min(cfg.folds,5) if not cfg.quick else 3
    for j,(tr,va) in enumerate(KFold(n_splits=folds,shuffle=True,random_state=cfg.seed+123).split(df)):
        enc=FoldEncoder().fit(df.iloc[tr][baseline]);xtr=enc.transform(df.iloc[tr][baseline]);xva=enc.transform(df.iloc[va][baseline]);
        est=make_estimator(cfg.seed+44+j,cfg.quick).fit(xtr,y[tr]);oof_y[va]=y[va]-est.predict(xva)
        im=SimpleImputer(strategy='median',keep_empty_features=True).fit(xmat[tr]);a=im.transform(xmat[tr]);b=im.transform(xmat[va]);
        # Continuous features have very different units. Standardize targets from training fold,
        # use multivariate ridge, and convert predicted values back to their native scales.
        avg=np.mean(a,axis=0);scale=np.std(a,axis=0);scale=np.where(scale<1e-8,1,scale)
        mod=Ridge(alpha=60).fit(xtr,(a-avg)/scale)
        oof_fx[va]=b-(mod.predict(xva)*scale+avg)
    rows=[]
    for j,c in enumerate(xcols):
        a=oof_fx[:,j];ok=np.isfinite(a)&np.isfinite(oof_y)
        if ok.sum()<30 or np.std(a[ok])<1e-9:continue
        rho,p=stats.spearmanr(a[ok],oof_y[ok])
        raw=df[c].to_numpy(dtype=float);ok0=np.isfinite(raw)&np.isfinite(y)
        rawrho=stats.spearmanr(raw[ok0],y[ok0]).statistic if ok0.sum()>=30 and np.std(raw[ok0])>1e-9 else np.nan
        rows.append({'feature':c,'family':feature_families([c])[c],
                     'n':int(ok.sum()),'raw_rho':float(rawrho),'conditional_rho':float(rho),
                     'conditional_p_unadjusted':float(p),
                     'conditional_median_abs_feature_residual':float(np.median(np.abs(a[ok]))),
                     'train_missing_fraction':float(df[c].isna().mean())})
    tab=pd.DataFrame(rows)
    if tab.empty:return tab,pd.DataFrame()
    tab['q_bh']=bh_adjust(tab.conditional_p_unadjusted)
    tab['abs_conditional_rho']=tab.conditional_rho.abs()
    tab=tab.sort_values('abs_conditional_rho',ascending=False)
    safe_save(tab,out/'conditional_feature_score_associations.csv')
    resid=pd.DataFrame({'subject_id':df.index,'y':y,'screening_oof_residual':oof_y})
    safe_save(resid,out/'screening_oof_residuals.csv')
    return tab,resid


def paired_student_bootstrap(pr,ref='screening_context',nboot=800,seed=2027,cluster_by_school=False):
    """Use repeat-averaged per-student OOF predictions; resample independent
    students OR school clusters. School resampling reflects school heterogeneity.
    """
    if pr.empty or ref not in set(pr.experiment):return pd.DataFrame()
    g=pr.groupby(['subject_id','school','experiment'],as_index=False).agg(y=('y','first'),pred=('pred','mean'))
    w=g.pivot(index=['subject_id','school','y'],columns='experiment',values='pred').reset_index()
    if ref not in w:return pd.DataFrame()
    rng=np.random.default_rng(seed)
    names=[c for c in w if c not in ('subject_id','school','y',ref)]
    n=len(w);schools=w.school.unique()
    rows=[]
    for name in names:
        a=w[['school','y',ref,name]].dropna()
        if len(a)<15:continue
        truth=a.y.to_numpy(float);b=a[ref].to_numpy(float);alt=a[name].to_numpy(float)
        r2delta=float(r2_score(truth,alt)-r2_score(truth,b))
        absdelta=float(mean_absolute_error(truth,alt)-mean_absolute_error(truth,b))
        delta=[]
        if cluster_by_school:
            groups=[np.where(a.school.to_numpy()==school)[0] for school in a.school.unique()]
            for _ in range(nboot):
                ids=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
                if len(np.unique(truth[ids]))<2:continue
                delta.append(r2_score(truth[ids],alt[ids])-r2_score(truth[ids],b[ids]))
        else:
            for _ in range(nboot):
                ids=rng.integers(0,len(a),len(a))
                if len(np.unique(truth[ids]))<2:continue
                delta.append(r2_score(truth[ids],alt[ids])-r2_score(truth[ids],b[ids]))
        rows.append({'comparison':name,'reference':ref,'resampling':'school_cluster' if cluster_by_school else 'student',
                     'n_students':len(a),'n_schools':a.school.nunique(),'delta_r2':r2delta,
                     'delta_mae':absdelta,'lo95':float(np.quantile(delta,.025)),
                     'hi95':float(np.quantile(delta,.975)),
                     'bootstrap_positive_fraction':float(np.mean(np.array(delta)>0))})
    return pd.DataFrame(rows)


def feature_shift_audit(df,te,out):
    rows=[]
    cols=[c for c in df.select_dtypes(include='number') if c in te]
    for c in cols:
        a=df[c].to_numpy(dtype=float);b=te[c].to_numpy(dtype=float)
        av=a[np.isfinite(a)];bv=b[np.isfinite(b)]
        if len(av)<20 or len(bv)<20:continue
        if np.std(av)<1e-12 and np.std(bv)<1e-12:continue
        res=stats.ks_2samp(av,bv,method='auto')
        iqr=np.subtract(*np.percentile(av,[75,25]));scale=iqr if iqr>1e-9 else np.std(av)+1e-8
        rows.append({'feature':c,'family':feature_families([c])[c],
                     'ks':float(res.statistic),'p_unadjusted':float(res.pvalue),
                     'median_shift_in_train_iqr':float((np.median(bv)-np.median(av))/scale),
                     'train_missing':float(np.mean(~np.isfinite(a))),
                     'test_missing':float(np.mean(~np.isfinite(b)))})
    t=pd.DataFrame(rows)
    if not t.empty:
        t['q_bh']=bh_adjust(t.p_unadjusted)
        t=t.sort_values('ks',ascending=False)
        safe_save(t,Path(out)/'feature_distribution_shift.csv')
    return t


def stability_and_confounding(df,y,schools,output,cfg:ResearchConfig):
    """Descriptive fit only. ARI measures perturbation stability, not phenotype validity."""
    if len(df)<30:return pd.DataFrame(),pd.DataFrame()
    enc=FoldEncoder().fit(df);X=enc.transform(df)
    baseline_oof=crossfit_baseline(X,y,Config(quick=cfg.quick,inner_folds=min(3,cfg.folds)),cfg.seed+88)
    ref=ResidualDiscovery(k=2,dim=3,lam=.5,seed=cfg.seed).fit(X,y,baseline_oof)
    _,q=ref.transform(X);ref_assign=q.argmax(axis=1)
    rng=np.random.default_rng(cfg.seed)
    rows=[]
    n=3 if cfg.quick else cfg.stability_resamples
    for j in range(n):
        # Subsample without replacement to prevent duplicated residual targets within
        # supervised PLS and avoid artificially high apparent stability.
        ids=rng.choice(len(y),size=max(25,int(.8*len(y))),replace=False)
        model=ResidualDiscovery(k=2,dim=3,lam=.5,seed=cfg.seed+j+1).fit(X[ids],y[ids],baseline_oof[ids])
        _,p=model.transform(X);assign=p.argmax(axis=1)
        rows.append({'seed':int(cfg.seed+j+1),'ari_vs_reference':float(adjusted_rand_score(ref_assign,assign)),
                     'min_cluster_fraction':float(np.bincount(assign,minlength=2).min()/len(assign)),
                     'mean_max_component_prob':float(p.max(axis=1).mean()),
                     'school_nmi':float(normalized_mutual_info_score(schools.astype(str),assign))})
    stab=pd.DataFrame(rows);safe_save(stab,Path(output)/'descriptive_group_stability.csv')
    pairs=pd.DataFrame({'subject_id':df.index,'group':ref_assign,'max_mixture_weight':q.max(axis=1),
                        'school':schools,'outcome':y,'screening_residual_oof':y-baseline_oof})
    safe_save(pairs,Path(output)/'descriptive_group_confounding.csv')
    return stab,pairs


def matched_behavior_pairs(df,y,out,max_pairs=150):
    """Match on screening covariates only. Score gaps analyzed after matching.
    Pure descriptive nearest-neighbor analysis, never part of predictive model.
    """
    cols=_pick_screening(df)
    if len(cols)<1:return pd.DataFrame()
    enc=FoldEncoder().fit(df[cols]);X=enc.transform(df[cols])
    neighbors=NearestNeighbors(n_neighbors=min(5,len(df)),algorithm='auto').fit(X)
    dist,idx=neighbors.kneighbors(X)
    used=set();rows=[];fam=_family_cols(df)
    for i in range(len(df)):
        for distance,j in zip(dist[i,1:],idx[i,1:]):
            key=tuple(sorted((i,int(j))))
            if key in used:continue
            used.add(key)
            rows.append({'subject_a':str(df.index[i]),'subject_b':str(df.index[j]),
                         'screening_distance':float(distance),
                         'score_gap_abs':float(abs(y[i]-y[j])),
                         'score_gap_signed':float(y[i]-y[j]),
                         'speech_feature_distance':float(np.nanmean(np.abs(df.iloc[i][fam.get('speech_text',[])].to_numpy(dtype=float)-df.iloc[j][fam.get('speech_text',[])].to_numpy(dtype=float)))) if fam.get('speech_text') else np.nan,
                         'acoustic_feature_distance':float(np.nanmean(np.abs(df.iloc[i][fam.get('audio',[])].to_numpy(dtype=float)-df.iloc[j][fam.get('audio',[])].to_numpy(dtype=float)))) if fam.get('audio') else np.nan})
            break
    tab=pd.DataFrame(rows).sort_values('screening_distance').head(max_pairs)
    safe_save(tab,Path(out)/'ability_matched_pairs.csv')
    return tab


def instrument_audit(df,oof,out):
    cols=[c for c in df.columns if 'instrument' in c.lower()]
    if not cols or oof.empty:return pd.DataFrame()
    instrument=df[cols[0]].astype(str)
    pred=oof[oof.repeat==oof.repeat.min()].copy()
    pred['instrument']=pred.subject_id.map(instrument)
    rows=[]
    for (name,ins),part in pred.groupby(['model','instrument']):
        if len(part)<4:continue
        rows.append({'instrument':ins,'model':name,'n':len(part),**score(part.y,part.pred),
                     'bias_pred_minus_truth':float((part.pred-part.y).mean())})
    tab=pd.DataFrame(rows)
    if not tab.empty:safe_save(tab,Path(out)/'instrument_oof.csv')
    return tab


def loso_model_deltas(out):
    """Within-school paired model gains; equal-school summary is separate from pooled R².

    One-student schools have undefined R²; MAE and bias remain valid.
    """
    path=Path(out)/'tables'/'loso_predictions.csv'
    if not path.exists():return pd.DataFrame()
    t=pd.read_csv(path,dtype={'subject_id':str,'school':str})
    if 'global' not in set(t.model):return pd.DataFrame()
    rows=[]
    for school,p in t.groupby('school'):
        wide=p.pivot(index='subject_id',columns='model',values='pred')
        yy=p.drop_duplicates('subject_id').set_index('subject_id')['y'].reindex(wide.index)
        if 'global' not in wide:continue
        base=np.abs(wide['global'].to_numpy()-yy.to_numpy())
        for name in [n for n in wide if n!='global']:
            err=np.abs(wide[name].to_numpy()-yy.to_numpy())
            rows.append({'school':school,'model':name,'n_students':len(p.subject_id.unique()),
                         'delta_mae_vs_global':float(err.mean()-base.mean()),
                         'share_students_improved':float((err<base).mean()),
                         'global_mae':float(base.mean()),'candidate_mae':float(err.mean())})
    tab=pd.DataFrame(rows)
    if not tab.empty:safe_save(tab,Path(out)/'tables'/'loso_school_model_deltas.csv')
    return tab


def descriptive_residual_rules(df,y,output,cfg:ResearchConfig):
    """Export readable 2-depth rules on training OOF screening residuals.

    EXPLORATORY: fitted to all labeled training students after computing OOF
    residuals; held-out validity is assessed separately in M8 nested CV.
    """
    from sklearn.tree import DecisionTreeRegressor,export_text
    screen=_pick_screening(df)
    cols=[c for c in df.select_dtypes(include='number') if c not in screen]
    if not cols:return pd.DataFrame()
    cols=cols[:max(1,cfg.max_assoc_features)]
    X=df[cols].replace([np.inf,-np.inf],np.nan)
    imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(X)
    feats=imp.transform(X)
    nfold=min(5,len(y))
    resid=np.full(len(y),np.nan)
    for j,(a,b) in enumerate(KFold(n_splits=nfold,shuffle=True,random_state=cfg.seed+456).split(df)):
        enc=FoldEncoder().fit(df.iloc[a][screen]);xa=enc.transform(df.iloc[a][screen]);xb=enc.transform(df.iloc[b][screen])
        model=make_estimator(cfg.seed+500+j,cfg.quick).fit(xa,y[a]);resid[b]=y[b]-model.predict(xb)
    tree=DecisionTreeRegressor(max_depth=2,min_samples_leaf=max(8,int(len(df)*.12)),
                               min_impurity_decrease=.05,random_state=cfg.seed).fit(feats,resid)
    leaf=tree.apply(feats)
    rows=[]
    for node in np.unique(leaf):
        idx=np.where(leaf==node)[0]
        rows.append({'leaf':int(node),'n':len(idx),'pct_train':len(idx)/len(y),
                     'mean_baseline_residual':float(resid[idx].mean()),
                     'sd_baseline_residual':float(resid[idx].std()),
                     'mean_target':float(y[idx].mean())})
    t=pd.DataFrame(rows)
    safe_save(t,Path(output)/'tables'/'descriptive_residual_rule_leaves.csv')
    rules=export_text(tree,feature_names=cols,decimals=3)
    (Path(output)/'reports'/'descriptive_residual_rules.txt').write_text(
        'Exploratory only: rule split on all labeled training features; not independent verification.\n\n'+rules)
    return t

def _plot(figdir,name,title,xlabel=None,ylabel=None):
    if xlabel:plt.xlabel(xlabel)
    if ylabel:plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(Path(figdir)/(name+'.png'),dpi=165,bbox_inches='tight')
    plt.close()


def figure_atlas(output,abl,ft,summary,assoc,resid,drift,stab,groups,pairs,bt,school_bt,inst,loso_deltas):
    folder=Path(output)/'figures';folder.mkdir(parents=True,exist_ok=True)
    if not summary.empty:
        fig,ax=plt.subplots(figsize=(10,max(4,len(summary)*.37)))
        d=summary.sort_values('mean_r2');ax.barh(d.experiment,d.mean_r2)
        ax.set_xlabel('Held-out fold R² (average; same student splits)')
        ax.set_title('E01 · Feature-family ablations');fig.tight_layout();fig.savefig(folder/'E01_feature_family_ablation.png',dpi=165,bbox_inches='tight');plt.close(fig)
        d=summary[summary.folds_compared>0].sort_values('mean_paired_delta_r2')
        if len(d):
            fig,ax=plt.subplots(figsize=(9,max(3,len(d)*.35)))
            ax.barh(d.experiment,d.mean_paired_delta_r2)
            ax.axvline(0,color='k',ls='--',alpha=.6);ax.set_xlabel('Mean paired fold ΔR² against declared reference')
            ax.set_title('E02 · Incremental value; paired folds, not independent observations')
            fig.tight_layout();fig.savefig(folder/'E02_fold_gain.png',dpi=165,bbox_inches='tight');plt.close(fig)
    if not assoc.empty:
        top=assoc.head(25).iloc[::-1]
        fig,ax=plt.subplots(figsize=(11,9))
        ax.barh(top.feature,top.conditional_rho)
        ax.axvline(0,color='k',lw=1);ax.set_xlabel('Rank association after cross-fitted screening adjustment')
        ax.set_title('E03 · Score residual vs feature residual (exploratory)')
        fig.tight_layout();fig.savefig(folder/'E03_conditional_associations.png',dpi=165,bbox_inches='tight');plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5));x=assoc.conditional_rho.to_numpy(float);p=assoc.q_bh.to_numpy(float)
        p=np.clip(p,1e-12,1);ax.scatter(x,-np.log10(p),c=['#18764c' if a<.1 else '#777777' for a in p],alpha=.65,s=18)
        ax.axvline(0,color='k',lw=.8);ax.axhline(-np.log10(.1),ls='--',color='#aa7722')
        ax.set_xlabel('Conditional Spearman ρ');ax.set_ylabel('−log₁₀(BH-adjusted q)')
        ax.set_title('E04 · Multiple-testing-aware conditional evidence')
        fig.tight_layout();fig.savefig(folder/'E04_association_fdr.png',dpi=165,bbox_inches='tight');plt.close(fig)
    if not bt.empty:
        d=bt[bt.resampling=='student'].sort_values('delta_r2').copy()
        if len(d):
            fig,ax=plt.subplots(figsize=(9,max(3,len(d)*.4)))
            errs=[np.maximum(0,d.delta_r2-d.lo95),np.maximum(0,d.hi95-d.delta_r2)]
            ax.errorbar(d.delta_r2,np.arange(len(d)),xerr=errs,fmt='o',capsize=3)
            ax.set_yticks(np.arange(len(d)),d.comparison);ax.axvline(0,color='k',ls='--')
            ax.set_xlabel('Student-bootstrap ΔR² vs screening + context')
            ax.set_title('E05 · Repeated-OOF paired confidence intervals')
            fig.tight_layout();fig.savefig(folder/'E05_student_bootstrap.png',dpi=165,bbox_inches='tight');plt.close(fig)
    if not school_bt.empty:
        d=school_bt.sort_values('delta_r2');fig,ax=plt.subplots(figsize=(9,max(3,len(d)*.4)))
        ax.errorbar(d.delta_r2,np.arange(len(d)),xerr=[np.maximum(0,d.delta_r2-d.lo95),np.maximum(0,d.hi95-d.delta_r2)],fmt='o',capsize=3)
        ax.set_yticks(np.arange(len(d)),d.comparison);ax.axvline(0,color='k',ls='--')
        ax.set_xlabel('School-cluster-bootstrap ΔR²');ax.set_title('E06 · School-aware uncertainty (unequal school sizes)')
        fig.tight_layout();fig.savefig(folder/'E06_school_bootstrap.png',dpi=165,bbox_inches='tight');plt.close(fig)
    if not drift.empty:
        d=drift.head(25).iloc[::-1];fig,ax=plt.subplots(figsize=(10,8));ax.barh(d.feature,d.ks)
        ax.set_xlabel('Train-test KS statistic (descriptive, no labels)');ax.set_title('E07 · Train/test feature distribution differences')
        fig.tight_layout();fig.savefig(folder/'E07_train_test_shift.png',dpi=165,bbox_inches='tight');plt.close(fig)
    if not stab.empty:
        fig,ax=plt.subplots(figsize=(7,4));ax.hist(stab.ari_vs_reference,bins=np.linspace(-.1,1,18),edgecolor='white');ax.set_xlim(-.1,1.05)
        ax.set_xlabel('ARI compared with original descriptive fit');ax.set_ylabel('Subsample runs')
        ax.set_title('E08 · Cluster perturbation stability — not predictive validity')
        fig.tight_layout();fig.savefig(folder/'E08_group_stability.png',dpi=165,bbox_inches='tight');plt.close(fig)
    if not groups.empty:
        fig,ax=plt.subplots(figsize=(7,4));gs=sorted(groups.group.unique());dd=[groups[groups.group==x].screening_residual_oof for x in gs]
        ax.boxplot(dd, tick_labels=[str(x) for x in gs]);ax.axhline(0,ls='--',color='k')
        ax.set_xlabel('Descriptive cluster label');ax.set_ylabel('OOF baseline residual')
        ax.set_title('E09 · Outcome-supervised descriptive group association')
        fig.tight_layout();fig.savefig(folder/'E09_group_residuals.png',dpi=165,bbox_inches='tight');plt.close(fig)
        gp=groups.groupby('group',as_index=False).agg(n=('group','size'),mean_residual=('screening_residual_oof','mean'),mean_certainty=('max_mixture_weight','mean'))
        gp.to_csv(Path(output)/'tables'/'descriptive_group_summary.csv',index=False)
    if not pairs.empty:
        fig,ax=plt.subplots(figsize=(7,4));ax.scatter(pairs.screening_distance,pairs.score_gap_abs,s=13,alpha=.6)
        ax.set_xlabel('Distance on screening/context covariates');ax.set_ylabel('Absolute later-score difference')
        ax.set_title('E10 · Similar screening profiles can diverge (descriptive)')
        fig.tight_layout();fig.savefig(folder/'E10_matched_heterogeneity.png',dpi=165,bbox_inches='tight');plt.close(fig)
    if not inst.empty:
        d=inst[inst.model=='global'].copy()
        if not d.empty:
            fig,ax=plt.subplots(figsize=(7,4));ax.bar(d.instrument,d.bias_pred_minus_truth)
            ax.axhline(0,color='k',ls='--');ax.set_ylabel('Mean (predicted−actual)')
            ax.set_title('E11 · Prediction bias by assessment instrument')
            fig.tight_layout();fig.savefig(folder/'E11_instrument_bias.png',dpi=165,bbox_inches='tight');plt.close(fig)
    if not abl.empty:
        d=abl[(abl.experiment=='all_features')&(abl.repeat==abl.repeat.min())].copy()
        if not d.empty:
            fig,ax=plt.subplots(figsize=(7,4));ax.scatter(d.pred,d.y-d.pred,s=15,alpha=.55);ax.axhline(0,color='k',ls='--')
            ax.set_xlabel('Full model OOF prediction');ax.set_ylabel('Observed−predicted')
            ax.set_title('E12 · Residual structure and heteroscedasticity')
            fig.tight_layout();fig.savefig(folder/'E12_residual_vs_prediction.png',dpi=165,bbox_inches='tight');plt.close(fig)
    if not loso_deltas.empty:
        d=loso_deltas.pivot(index='school',columns='model',values='delta_mae_vs_global')
        ax=d.plot.bar(figsize=(max(9,len(d)*.5),5),rot=45)
        ax.axhline(0,color='k',ls='--')
        ax.set_ylabel('LOSO ΔMAE vs global (negative is better)')
        ax.set_title('E13 · Unseen-school paired local-model gains')
        plt.tight_layout();plt.savefig(folder/'E13_loso_school_deltas.png',dpi=165,bbox_inches='tight');plt.close()
    # Always include an index so a reader can navigate without rerunning cells.
    imgs=sorted(folder.glob('E*.png'))
    html=['<!doctype html><html><head><meta charset="utf-8"><title>WildSpeech M8.1 research atlas</title>',
          '<style>body{font:16px system-ui;margin:2em auto;max-width:1150px;color:#1b2430;background:#f8fafc}h1,h2{color:#122a47}section{background:#fff;padding:20px;margin:22px 0;border:1px solid #e2e8f0;border-radius:12px}img{max-width:100%;height:auto}p{line-height:1.55}</style></head><body>',
          '<h1>WildSpeech M8.1 · Enhanced Experiments and Analysis</h1>',
          '<p>All figures generated from the current run. Group fits are descriptive unless explicitly cross-validated. Conditional correlations are not causal.</p>']
    for p in imgs:html.append(f'<section><h2>{p.stem.replace("_"," ")}</h2><img src="figures/{p.name}" loading="lazy" alt="{p.stem}"></section>')
    html.append('</body></html>')
    (Path(output)/'RESEARCH_ATLAS.html').write_text('\n'.join(html),encoding='utf8')


def research_summary(output,summary,assoc,bt,schoolbt,stab,drift):
    def fmt(x):return 'n/a' if x is None or not np.isfinite(x) else f'{x:.4f}'
    lines=['# WildSpeech M8.1 — Evidence report','',
           '> All numerical findings below come from the *current run*. This is not a causal analysis.','',
           '## Interpretation priorities','',
           '1. **Incremental signal:** read feature-family ablations against screening/context on the same folds.',
           '2. **Uncertainty:** check both student and school-cluster bootstrap confidence intervals.',
           '3. **Speech controls:** compare full features to within-fold shuffled acoustic/speech features.',
           '4. **Conditional associations:** q-values are FDR-adjusted, but exploratory post-selection and correlated features remain concerns.',
           '5. **Groups:** stability is not proof of distinct literacy phenotypes.',
           '6. **No test outcome usage:** train-test shift diagnostics use test covariates only.','']
    if not summary.empty:
        lines+=['## Feature-family ablation','',summary[['experiment','features','mean_r2','mean_paired_delta_r2','folds_better','folds_compared']].to_markdown(index=False,floatfmt='.4f'),'']
    if not bt.empty:
        lines+=['## Student bootstrap (repeat-averaged OOF)','',bt[['comparison','delta_r2','lo95','hi95']].to_markdown(index=False,floatfmt='.4f'),'']
    if not schoolbt.empty:
        lines+=['## School-cluster bootstrap','',schoolbt[['comparison','delta_r2','lo95','hi95']].to_markdown(index=False,floatfmt='.4f'),'']
    if not assoc.empty:
        lines+=['## Strongest conditional feature associations (exploratory)','',assoc[['feature','family','conditional_rho','q_bh']].head(15).to_markdown(index=False,floatfmt='.4f'),'']
    if not stab.empty:
        lines+=['## Descriptive group stability','',f'ARI median={fmt(stab.ari_vs_reference.median())}; IQR={fmt(stab.ari_vs_reference.quantile(.25))}–{fmt(stab.ari_vs_reference.quantile(.75))}.','']
    if not drift.empty:
        lines+=['## Largest train–test feature shifts','',drift[['feature','family','ks','q_bh']].head(10).to_markdown(index=False,floatfmt='.4f'),'']
    if not summary.empty and 'all_features' in set(summary.experiment):
        r=summary.set_index('experiment').loc['all_features']
        if np.isfinite(r.mean_paired_delta_r2):
            lines+=['## Automated diagnostic interpretation','',
                    f'All-features vs screening/context: mean paired-fold ΔR² = {fmt(r.mean_paired_delta_r2)} '
                    f'(improved {int(r.folds_better)}/{int(r.folds_compared)} folds).']
            if r.mean_paired_delta_r2<=0:
                lines+=['**Current ablation does not support expanding the global baseline.**']
            else:
                lines+=['The incremental result is positive but requires examination of both bootstrap intervals and independent transfer data.']
    if not assoc.empty:
        significant=assoc[assoc.q_bh<.1]
        lines+=[f'Exploratory conditional features passing BH q<0.10: {len(significant)} / {len(assoc)}.']
    if not stab.empty:
        lines+=[f'Descriptive grouping: median bootstrap ARI {fmt(stab.ari_vs_reference.median())}; '
                f'mean school-NMI {fmt(stab.school_nmi.mean())}. Neither establishes causal or predictive group validity.']
    lines+=['']
    lines+=['## Recommended decision rules','',
            '- Treat the global model as the default until the full feature stack shows reliable incremental benefit on independently locked holdouts.',
            '- Treat small ΔR² with intervals crossing zero as exploratory.',
            '- Do not interpret unsupervised distances or GMM membership weights as validated reading-skill types.',
            '- Keep phonetic hypotheses separate from validated child speech and adult-contamination labels.',
            '- Do not submit a model based only on the best result among many observed outer-fold comparisons.','']
    (Path(output)/'RESEARCH_SUMMARY.md').write_text('\n'.join(lines),encoding='utf8')


def run_research(cfg:Config,research:ResearchConfig|None=None):
    rc=research or ResearchConfig(seed=cfg.seed+1,quick=cfg.quick)
    path=Path(cfg.output_root);table=path/'tables';table.mkdir(parents=True,exist_ok=True)
    tr,te,y,schools,fmt,_,manifest=prepare_dataset(cfg)
    print(f'[M8.1] n_train={len(tr)}, n_test={len(te)}, features={tr.shape[1]}, school_count={len(set(schools))}',flush=True)
    a,folds,summary=experiment_ablation(tr,y,schools,table,rc)
    assoc,resid=conditional_associations(tr,y,table,rc)
    drift=feature_shift_audit(tr,te,table)
    stab,groups=stability_and_confounding(tr,y,np.asarray(schools),table,rc)
    pairs=matched_behavior_pairs(tr,y,table)
    bt=paired_student_bootstrap(a,nboot=rc.bootstrap,seed=rc.seed)
    schoolbt=paired_student_bootstrap(a,nboot=rc.bootstrap,seed=rc.seed+10,cluster_by_school=True)
    if not bt.empty:safe_save(bt,table/'paired_student_bootstrap_ablation.csv')
    if not schoolbt.empty:safe_save(schoolbt,table/'paired_school_bootstrap_ablation.csv')
    original_pred=table/'student_oof_predictions.csv';inst=pd.DataFrame()
    if original_pred.exists():
        original=pd.read_csv(original_pred,dtype={'subject_id':str})
        first=original[(original.model=='global')&(original.repeat==original.repeat.min())]
        if len(first)==len(tr) and first.subject_id.is_unique:
            inst=instrument_audit(tr,original,table)
    loso=loso_model_deltas(path)
    descriptive_residual_rules(tr,y,path,rc)
    figure_atlas(path,a,folds,summary,assoc,resid,drift,stab,groups,pairs,bt,schoolbt,inst,loso)
    research_summary(path,summary,assoc,bt,schoolbt,stab,drift)
    result={'research_version':VERSION,'train':len(tr),'test':len(te),'n_features':len(tr.columns),
            'num_figures':len(list((path/'figures').glob('E*.png'))),
            'ablation_experiments':list(summary.experiment) if not summary.empty else [],
            'mean_r2_full':float(summary.set_index('experiment').loc['all_features','mean_r2']) if not summary.empty else None,
            'research_note':'Synthetic tests do not measure genuine competition performance'}
    report_json(result,path/'reports'/'m81_research_summary.json')
    print(f'[M8.1 DONE] figures={result["num_figures"]} atlas={path/"RESEARCH_ATLAS.html"}',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--profile-csv',default='')
    p.add_argument('--output-root',required=True);p.add_argument('--quick',action='store_true')
    p.add_argument('--folds',type=int,default=5);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--permutations',type=int,default=3);p.add_argument('--bootstrap',type=int,default=800)
    p.add_argument('--stability',type=int,default=25);p.add_argument('--max-features',type=int,default=450)
    a=p.parse_args();cfg=Config(data_root=a.data_root,profile_csv=a.profile_csv,output_root=a.output_root,
                                  quick=a.quick,max_features=a.max_features)
    run_research(cfg,ResearchConfig(folds=a.folds,repeats=a.repeats,permutations=a.permutations,
                                    bootstrap=a.bootstrap,stability_resamples=a.stability,quick=a.quick))

if __name__=='__main__':main()
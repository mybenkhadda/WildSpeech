"""WildSpeech M8.2: reproducible, feature-only pairwise consensus research.

Entry point: python -m wildspeech_m8.m82_pipeline --mode FULL
Reuses M8.1 cached student/audio/phonetic feature tables. It does not
redownload/re-extract audio, alter M8.1 submissions, or use test outcomes.
"""
from __future__ import annotations
import argparse,json,time
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from .m8_pipeline import Config,prepare_dataset,read_official,FoldEncoder,make_estimator
from .m82_views import ViewBuilder
from .m82_consensus import ConsensusConfig,consensus_ensemble
from .m82_pair_analysis import (attach_pair_diagnostics,summarize_pairs,comparison_descriptive,
                                stable_graph_nodes,consensus_score_relationship,school_confound_summary)
from .m82_figures import generate_atlas
from .m82_prediction import PredictionConfig,evaluate_predictive,paired_bootstrap

@dataclass
class M82Config:
    data_root:str='/content/drive/MyDrive/WildSpeech'
    output_root:str='/content/drive/MyDrive/WildSpeech_M8'
    profile_csv:str=''
    mode:str='SMOKE'
    max_features:int=450
    max_per_view:int=36
    resamples:int=5
    row_fraction:float=.82
    min_pair_coverage:float=.38
    seed:int=2026
    run_predictive:bool=True
    predictive_folds:int=5
    predictive_repeats:int=1
    use_existing_screening_oof:bool=False # default recompute, prevents stale cache


def screening_oof(frame,y,seed,quick):
    from .m8_research import feature_sets
    sets=feature_sets(frame)
    cols=sets.get('screening_context') or sets['screening_only']
    if not cols:raise ValueError('No screening features for OOF residuals')
    out=np.full(len(frame),np.nan)
    folds=min(3 if quick else 5,len(frame)//5)
    if folds<2:raise ValueError('Not enough training students to crossfit residuals')
    for j,(a,b) in enumerate(KFold(n_splits=folds,shuffle=True,random_state=seed).split(frame)):
        enc=FoldEncoder().fit(frame.iloc[a][cols])
        model=make_estimator(seed+j,fast=True).fit(enc.transform(frame.iloc[a][cols]),y[a])
        out[b]=y[b]-model.predict(enc.transform(frame.iloc[b][cols]))
    if not np.isfinite(out).all():raise AssertionError('Incomplete screening OOF residuals')
    return out


def _write(df,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False)


def run_feature_frame(frame:pd.DataFrame,y,schools,output_root:str|Path,cfg:M82Config,
                      residual=None,instruments=None)->dict:
    """Analysis from a validated student feature frame. Used by tests and official run."""
    start=time.monotonic()
    out=Path(output_root);(out/'tables').mkdir(parents=True,exist_ok=True)
    (out/'figures').mkdir(parents=True,exist_ok=True)
    (out/'reports').mkdir(parents=True,exist_ok=True)
    if not frame.index.is_unique:raise ValueError('Student index not unique')
    ids=frame.index.astype(str).to_numpy();n=len(ids)
    schools=np.asarray(schools)
    y=np.asarray(y,float)
    if len(y)!=n or len(schools)!=n or not np.isfinite(y).all():raise ValueError('Training student/outcome alignment error')
    quick=cfg.mode.upper()=='SMOKE'
    if residual is None:
        print('[M8.2] Cross-fitting screening-only prediction residuals',flush=True)
        residual=screening_oof(frame,y,cfg.seed+5,quick)
    else:
        residual=np.asarray(residual,float)
    if len(residual)!=n or not np.isfinite(residual).all():raise ValueError('Residual size/finite validation failed')
    builder=ViewBuilder(max_per_view=cfg.max_per_view,min_nonmissing=.5 if quick else .55)
    views=builder.fit_transform(frame,include_nuisance=True)
    nuisance=views.pop('nuisance_audit',None)
    if len(views)<2:raise RuntimeError(f'At least two independent feature views required, found {list(views)}')
    # For quick diagnostics only fewer rounds, same explicit minimum support.
    cc=ConsensusConfig(seed=cfg.seed,row_fraction=cfg.row_fraction,
                       resamples=min(2,cfg.resamples) if quick else cfg.resamples,
                       min_pair_coverage=cfg.min_pair_coverage,
                       cluster_counts=(2,3) if quick else (2,3,4,6,8),
                       algorithms=('kmeans','ward') if quick else ('kmeans','gmm','ward'),
                       min_pair_runs=1 if quick else max(3,cfg.resamples//2),
                       minimum_views=2)
    print(f'[M8.2] n_students={n} possible_pairs={n*(n-1)//2} views={list(views)}',flush=True)
    result=consensus_ensemble(views,cc)
    pair=attach_pair_diagnostics(result.pairs,views,ids,schools,y,residual,instruments)
    # Nuisance audit is measured but NEVER enters the consensus or prediction.
    if nuisance is not None:
        a=pair.i.to_numpy(int);b=pair.j.to_numpy(int)
        obs=nuisance.observed[a]&nuisance.observed[b];nobs=obs.sum(axis=1)
        delta=((nuisance.matrix[a]-nuisance.matrix[b])**2*obs).sum(axis=1)
        pair['nuisance_distance']=np.sqrt(np.divide(delta,nobs,out=np.full(len(pair),np.nan),where=nobs>0))
    # The atlas must see the enriched pair table, not the outcome-blind raw result.
    # Consensus construction above remains independent of y and residuals.
    result.pairs=pair
    pair['same_student']=False
    if (pair.subject_a==pair.subject_b).any():raise AssertionError('Self-pair found after export')
    if pair[['i','j']].duplicated().any():raise AssertionError('Duplicate pair found')
    # Preserve every pair in a compressed, columnar format; safe for ~146k pairs.
    try:
        pair.to_parquet(out/'tables'/'all_student_pairs.parquet',index=False)
    except (ImportError,ModuleNotFoundError):
        # Optional parquet engine missing: never lose the full pair dataset.
        pair.to_csv(out/'tables'/'all_student_pairs.csv.gz',index=False,compression='gzip')
    # Human-reviewable top pairs and disagreement cases; full table remains parquet.
    pair.sort_values('consensus',ascending=False,na_position='last').head(300).to_csv(out/'tables'/'top_consensus_pairs.csv',index=False)
    stable=pair[pair.category=='stable_together'].sort_values('screening_residual_gap',ascending=False)
    _write(stable.head(200),out/'tables'/'stable_but_outcome_discordant.csv')
    diff=pair[pair.category=='cross_view_disagreement'].sort_values('view_disagreement',ascending=False)
    _write(diff.head(200),out/'tables'/'view_disagreement_pairs.csv')
    summaries=summarize_pairs(pair); matched=comparison_descriptive(pair)
    graph=stable_graph_nodes(pair,n)
    graph['subject_id']=ids
    _write(graph,out/'tables'/'student_graph_components.csv')
    _write(summaries,out/'tables'/'pair_category_summary.csv')
    _write(matched,out/'tables'/'ability_matched_pair_summary.csv')
    _write(school_confound_summary(pair),out/'tables'/'school_confound_summary.csv')
    _write(consensus_score_relationship(pair),out/'tables'/'consensus_score_relationship.csv')
    _write(result.summary,out/'tables'/'view_ensemble_summary.csv')
    _write(result.runs,out/'tables'/'clustering_run_log.csv')
    view_manifest=[{'view':name,'n_features':len(v.columns),'columns':v.columns,
                    'mean_observed_coverage':float(v.observed.mean())} for name,v in views.items()]
    view_manifest.append({'view':'nuisance_audit_excluded','columns':nuisance.columns if nuisance is not None else [],
                          'n_features':len(nuisance.columns) if nuisance is not None else 0,
                          'mean_observed_coverage':float(nuisance.observed.mean()) if nuisance is not None else None})
    (out/'reports'/'view_manifest.json').write_text(json.dumps(view_manifest,indent=2))
    np.savez_compressed(out/'tables'/'consensus_matrices.npz',**{
        f'{name}__{key}':matrix for name,d in result.matrices.items() for key,matrix in d.items()
        if key in ('consensus','null_expected','support')})
    pred=None;pv=None
    if cfg.run_predictive:
        print('[M8.2] Testing inductive prototype residual model vs identical global baseline',flush=True)
        pred,pv=evaluate_predictive(frame,y,schools,PredictionConfig(
            folds=cfg.predictive_folds,repeats=cfg.predictive_repeats,seed=cfg.seed+100,
            quick=quick))
        _write(pred,out/'tables'/'pairwise_predictive_oof.csv')
        _write(pv,out/'tables'/'pairwise_predictive_fold_metrics.csv')
        _write(paired_bootstrap(pred,100 if quick else 500,cfg.seed+7),out/'tables'/'pairwise_predictive_student_bootstrap.csv')
        _write(paired_bootstrap(pred,100 if quick else 500,cfg.seed+8,school_cluster=True),out/'tables'/'pairwise_predictive_school_bootstrap.csv')
    figures=generate_atlas(result,out/'figures',score_summary=summaries,predictive=pv,component_nodes=graph)
    overview={'version':'M8.2','mode':cfg.mode,'students':n,'pairs':len(pair),'views':list(views),
              'nuisance_excluded_columns':len(nuisance.columns) if nuisance is not None else 0,
              'pair_categories':pair.category.value_counts().to_dict(),
              'figures':figures,'run_seconds':round(time.monotonic()-start,1),
              'prediction_tested':bool(pred is not None),
              'limitations':['Outcome scores not used to construct consensus',
                             'Pair observations are dependent, not independent samples',
                             'Null expected consensus is chance correction, not p-value',
                             'No audio re-extraction; phonetic view requires existing feature cache',
                             'Full unvalidated study; figures are descriptive; validate out of school']}
    (out/'reports'/'m82_run_summary.json').write_text(json.dumps(overview,indent=2))
    lines=['# M8.2 Pairwise consensus evidence report','',f'Analyzed **{n} students** and **{len(pair)} unique student pairs**.',
           '', '## View inventory','']
    lines.extend(f'- **{a["view"]}**: {a["n_features"]} features, mean available {a["mean_observed_coverage"]:.1%}' for a in view_manifest if a['mean_observed_coverage'] is not None)
    lines.extend(['','## Pair categories','',summaries.to_markdown(index=False,floatfmt='.3f'),
                  '','## Interpretation safeguards','',*('- '+x for x in overview['limitations'])])
    if pv is not None:
        lines.extend(['','## Outer-fold predictive comparator','',pv.groupby('model')[['r2','mae','rmse']].mean().to_markdown(floatfmt='.4f'),
                      '','The correction is fixed-strength and trained on **OOF residuals from training students only**. A gain must replicate before deployment.'])
    (out/'M82_RESEARCH_SUMMARY.md').write_text('\n'.join(lines),encoding='utf8')
    print('[M8.2 DONE]',overview['pairs'],'pairs',len(figures),'figures','in',out,flush=True)
    return overview


def run_m82(cfg:M82Config):
    cfg.mode=cfg.mode.upper()
    if cfg.mode not in ('SMOKE','FULL'):raise ValueError('mode must be SMOKE or FULL')
    base=Config(data_root=cfg.data_root,output_root=cfg.output_root,
                profile_csv=cfg.profile_csv,quick=cfg.mode=='SMOKE',max_features=cfg.max_features)
    frame,te,y,schools,fmt,clips,manifest=prepare_dataset(base)
    instruments=None
    for col in frame.columns:
        if 'instrument' in col.lower():
            instruments=frame[col].fillna('UNKNOWN').to_numpy();break
    out=Path(cfg.output_root)/'m82'
    result=run_feature_frame(frame,y,schools,out,cfg,instruments=instruments)
    result['n_test_students']=len(te)
    return result


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=None,help='Optional YAML configuration')
    p.add_argument('--data-root',default=None)
    p.add_argument('--output-root',default=None)
    p.add_argument('--profile-csv',default=None)
    p.add_argument('--mode',choices=['SMOKE','FULL'],default=None)
    p.add_argument('--resamples',type=int,default=None)
    p.add_argument('--max-features',type=int,default=None)
    p.add_argument('--skip-prediction',action='store_true')
    p.add_argument('--seed',type=int,default=None)
    a=p.parse_args()
    options={}
    if a.config:
        import yaml
        options=yaml.safe_load(Path(a.config).read_text()) or {}
        unknown=set(options)-set(M82Config.__dataclass_fields__)
        if unknown:raise ValueError(f'Unknown M8.2 YAML config fields: {sorted(unknown)}')
    for key in ('data_root','output_root','profile_csv','mode','resamples','max_features','seed'):
        val=getattr(a,key)
        if val is not None:options[key]=val
    if a.skip_prediction:options['run_predictive']=False
    run_m82(M82Config(**options))

if __name__=='__main__':cli()
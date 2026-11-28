"""M8.2 reproducible scientific figure atlas (training-only descriptive evidence)."""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,leaves_list
from scipy.spatial.distance import squareform


def generate_atlas(result,out,score_summary=None,predictive=None,component_nodes=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p=result.pairs;mats=result.matrices
    n=int(max(p.i.max(),p.j.max())+1)
    names=[v for v in mats if np.isfinite(mats[v].get('consensus',np.array([np.nan]))).any()]
    if not names:return []
    # Fill unsupported pairs with their null expectation rather than false 0/1.
    global_sum=np.zeros((n,n),float);counts=np.zeros((n,n),float)
    for k in names:
        a=mats[k]['consensus'];b=mats[k]['null_expected']
        ok=np.isfinite(a)
        global_sum[ok]+=a[ok];counts[ok]+=1
    overall=np.divide(global_sum,counts,out=np.full((n,n),np.nan),where=counts>0)
    chance=np.nanmean([np.nanmean(mats[k]['null_expected']) for k in names])
    overall=np.where(np.isfinite(overall),overall,chance)
    overall=(overall+overall.T)/2;np.fill_diagonal(overall,1)
    dist=np.clip(1-overall,0,1);np.fill_diagonal(dist,0.)
    order=leaves_list(linkage(squareform(dist,checks=False),method='average')) if n>2 else np.arange(n)
    made=[]
    def save(name):
        plt.tight_layout();plt.savefig(out/name,dpi=155,bbox_inches='tight');plt.close();made.append(name)
    plt.figure(figsize=(9,8));plt.imshow(overall[np.ix_(order,order)],vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
    plt.colorbar(label='Mean co-clustering probability across eligible views')
    plt.xlabel('Students reordered by consensus');plt.ylabel('Students')
    plt.title('M82-01 · Multi-view consensus (outcome-blind)');save('M82_01_consensus_heatmap.png')
    for idx,name in enumerate(names):
        if idx>=8:break
        a=mats[name]['consensus'][np.ix_(order,order)]
        plt.figure(figsize=(7,6));plt.imshow(np.ma.masked_invalid(a),vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
        plt.colorbar(label='Pair co-membership, eligible runs only')
        plt.title('M82-02 · '+name+' view');plt.xlabel('Students, consensus ordering');plt.ylabel('Students')
        save(f'M82_02_view_{idx:02d}_{name}.png')
    plt.figure(figsize=(8,4));q=p[['consensus','chance_expected']].dropna()
    if not q.empty:
        plt.hist(q.consensus,bins=np.linspace(0,1,31),alpha=.65,label='Observed')
        plt.hist(q.chance_expected,bins=np.linspace(0,1,31),alpha=.5,label='Cluster-size null expectation')
    plt.legend();plt.xlabel('Co-membership probability');plt.ylabel('Pairs');plt.title('M82-03 · Consensus versus chance');save('M82_03_consensus_vs_chance.png')
    cat=p.category.value_counts();plt.figure(figsize=(10,4));plt.bar(cat.index,cat.values)
    plt.xticks(rotation=20,ha='right');plt.ylabel('Pair count');plt.title('M82-04 · Robust relationship categories')
    save('M82_04_pair_categories.png')
    plt.figure(figsize=(8,4));supports=p[[x for x in p if x.startswith('support__')]]
    for col in supports.columns:
        plt.hist(supports[col],bins=20,histtype='step',label=col.replace('support__',''))
    plt.xlabel('Valid clustering runs for pair');plt.ylabel('Pairs');plt.title('M82-05 · Pair observability by modality')
    plt.legend(fontsize=8);save('M82_05_pair_coverage.png')
    vcols=[f'co__{x}' for x in names];cor=np.full((len(vcols),len(vcols)),np.nan)
    for ix,a in enumerate(vcols):
        for jx,b in enumerate(vcols):
            if a==b:
                cor[ix,jx]=1.
                continue
            v=p[[a,b]].dropna()
            cor[ix,jx]=np.corrcoef(v[a],v[b])[0,1] if len(v)>20 and v[a].std()>1e-8 and v[b].std()>1e-8 else np.nan
    plt.figure(figsize=(max(5,len(names)*1.1),max(4,len(names))));plt.imshow(np.ma.masked_invalid(cor),vmin=-1,vmax=1,cmap='coolwarm')
    plt.xticks(range(len(names)),names,rotation=40,ha='right');plt.yticks(range(len(names)),names)
    plt.colorbar(label='Correlation of pair consensus across views')
    plt.title('M82-06 · Cross-view agreement');save('M82_06_view_agreement.png')
    if 'score_gap' in p:
        rng=np.random.default_rng(42);avail=p[['consensus','score_gap','screening_residual_gap']].dropna() if 'screening_residual_gap' in p else p[['consensus','score_gap']].dropna()
        if len(avail):
            draw=avail.iloc[rng.choice(len(avail),size=min(12000,len(avail)),replace=False)]
            for j,target in enumerate(['score_gap','screening_residual_gap']):
                if target not in draw:continue
                plt.figure(figsize=(8,5));plt.scatter(draw.consensus,draw[target],s=3,alpha=.11)
                bins=pd.cut(avail.consensus,bins=np.linspace(0,1,11),include_lowest=True)
                tab=avail.groupby(bins,observed=True).agg(consensus=('consensus','mean'),gap=(target,'mean'),n=(target,'size'))
                if len(tab):plt.plot(tab.consensus,tab.gap,marker='o',lw=2,label='Binned mean (descriptive)');plt.legend()
                plt.xlabel('Outcome-blind pair consensus');plt.ylabel('Absolute '+target)
                plt.title(f'M82-0{7+j} · Pair consensus vs {target} (pairs dependent)');save(f'M82_0{7+j}_consensus_{target}.png')
        vals=[p.loc[p.category==c,'screening_residual_gap'].dropna().to_numpy() for c in cat.index if 'screening_residual_gap' in p]
        if vals and any(len(z)>0 for z in vals):
            cats=[c for c in cat.index if len(p.loc[p.category==c,'screening_residual_gap'].dropna())]
            plt.figure(figsize=(10,5));plt.boxplot([z[:3000] for z in vals if len(z)],showfliers=False)
            plt.xticks(np.arange(1,len(cats)+1),cats)
            plt.xticks(rotation=20,ha='right');plt.ylabel('OOF baseline residual gap')
            plt.title('M82-09 · Residual heterogeneity by pair category')
            save('M82_09_pair_residual_gap.png')
    if 'same_school' in p:
        plt.figure(figsize=(8,4));p.groupby(['same_school','category']).size().unstack(fill_value=0).T.plot.bar(ax=plt.gca())
        plt.title('M82-10 · Pair types by school match');plt.ylabel('Pairs');plt.xticks(rotation=20,ha='right')
        save('M82_10_school_pair_confound.png')
    if component_nodes is not None and len(component_nodes):
        vals=component_nodes.loc[component_nodes.component_size>1].drop_duplicates('component').component_size
        plt.figure(figsize=(8,4));plt.hist(vals,bins=range(2,int(max(3,vals.max() if len(vals) else 3))+2))
        plt.xlabel('Connected component size');plt.ylabel('Communities')
        plt.title('M82-11 · Stable-pair graph components, no forced k');save('M82_11_stable_graph_components.png')
    if predictive is not None and len(predictive):
        pv=predictive.groupby('model').agg(r2=('r2','mean'),mae=('mae','mean')).sort_values('r2')
        plt.figure(figsize=(9,4));plt.barh(pv.index,pv.r2);plt.xlabel('Mean outer-fold R²');plt.title('M82-12 · Leakage-safe predictive comparison')
        save('M82_12_predictive_cv.png')
    html=['<!DOCTYPE html><html lang="en"><meta charset="utf-8"><title>WildSpeech M8.2 Atlas</title>',
          '<style>body{max-width:1050px;margin:auto;font:16px system-ui;line-height:1.5;padding:30px;color:#202b35}img{max-width:100%;border:1px solid #ddd;border-radius:8px}section{margin:32px 0}</style>',
          '<h1>WildSpeech M8.2 · Pairwise consensus atlas</h1><p>Descriptive training-data research. No figure establishes causality or independently validated phenotypes.</p>']
    for name in made:
        html.append(f'<section><h2>{escape(name)}</h2><img loading="lazy" src="figures/{escape(name)}"></section>')
    html.append('</html>')
    # Atlas is at output root, while PNG files are in output/figures.
    (out.parent/'M82_RESEARCH_ATLAS.html').write_text('\n'.join(html),encoding='utf8')
    return made
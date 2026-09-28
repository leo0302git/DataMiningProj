"""Second-round evidence: development diagnostics, paired comparisons and audit."""
import json
import hashlib
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.eda import load_abalone
from experiments.explore_abalone import OUT, networks, all_heads
from experiments.run_abalone import scores
from models.rrl.abalone import prepare, RRL
from models.rrl.abalone_refine import predict_graph, predict_refit


def main():
    dev=pd.read_csv(OUT/'development.csv')
    if (OUT/'extension.csv').exists():
        dev=pd.concat([dev,pd.read_csv(OUT/'extension.csv')],ignore_index=True)
    means=dev.groupby(['network','head','kind']).agg(rmse=('rmse','mean'),train_rmse=('train_rmse','mean'),
              seed_std=('rmse','std')).reset_index().sort_values('rmse')
    means.to_csv(OUT/'development_summary.csv',index=False)
    rows=[]
    for nid in sorted(dev.network.unique()):
        group=means[means.network==nid]
        original=float(group[group.kind=='adam'].iloc[0].rmse)
        for kind in ['rules','linear_rules','hinge_rules','hinge_only']:
            if not (group.kind==kind).any():
                continue
            best=group[group.kind==kind].iloc[0]
            rows.append(dict(network=nid,kind=kind,head=int(best['head']),adam_rmse=original,
                             refit_rmse=best.rmse,delta=best.rmse-original))
    pd.DataFrame(rows).to_csv(OUT/'head_diagnostics.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,nid in zip(axes.flat,[0,3,4,9,10,12]):
        for seed in [314,2718]:
            curve=pd.read_json(OUT/f'curve_{nid}_{seed}.json')
            ax.plot(curve.epoch,curve.train_rmse,ls='--',label=f'{seed} train')
            ax.plot(curve.epoch,curve.validation_rmse,label=f'{seed} validation')
        ax.set(title=f'Network {nid}: {networks()[nid]["structure"]}',xlabel='Epoch',ylabel='RMSE (Adam head)')
        ax.legend(fontsize=7)
    fig.savefig(OUT/'learning_curves.png',dpi=150);plt.close(fig)
    if not all((OUT/f'all_metrics_{i}.json').exists() for i in range(5)):
        return
    frame=load_abalone()
    design=json.loads((OUT/'design.json').read_text())
    assert design['data_sha256']==hashlib.sha256((ROOT/'data/raw/abalone/abalone.data').read_bytes()).hexdigest()
    assert design['networks']==networks()
    folds=list(KFold(5,shuffle=True,random_state=0).split(frame))
    selected=json.loads((OUT/'shortlist.json').read_text())
    collected=[]; max_error=0.
    for family in ['all','pure','hinge_only']:
        indices=[]
        for fold,(tr,te) in enumerate(folds):
            pred=pd.read_csv(OUT/f'{family}_fold_{fold}.csv')
            assert np.array_equal(pred['index'],te)
            assert np.array_equal(pred.Rings,frame.iloc[te].Rings)
            indices.extend(pred['index'])
            row=json.loads((OUT/f'{family}_metrics_{fold}.json').read_text())
            search=pd.read_csv(OUT/f'inner_{fold}.csv')
            assert len(search)==3*len(selected)
            assert not search.duplicated(['candidate','inner_fold']).any()
            allowed={'all':['adam','rules','linear_rules','hinge_rules'],'pure':['adam','rules'],'hinge_only':['hinge_only']}[family]
            ids=[i for i,c in enumerate(selected) if all_heads()[c['head']]['kind'] in allowed]
            expected=int(search.groupby('candidate').rmse.mean().loc[ids].idxmin())
            assert row['candidate']==expected
            graph=json.loads((OUT/f'{family}_fold_{fold}.json').read_text())
            _,state=prepare(frame.iloc[tr],transform=graph['preprocessing']['transform'])
            for key in ['features','categories','mean','std','transform']:
                assert state[key]==graph['preprocessing'][key]
            err=float(np.max(np.abs(predict_graph(graph,frame.iloc[te])-pred.prediction)))
            assert err<1e-4
            max_error=max(max_error,err)
            for key,value in scores(pred.Rings.to_numpy(),pred.prediction.to_numpy()).items():
                assert np.isclose(row[key],value)
            collected.append(dict(model=f'exploration_{family}',**row))
        assert sorted(indices)==list(range(len(frame)))
    for name in ['ridge','forest','rrl_original','rrl_optimized']:
        for fold in range(5):
            row=json.loads((OUT.parent/name/f'fold_{fold}_metrics.json').read_text())
            collected.append(dict(model=name,**{k:v for k,v in row.items() if k!='config'}))
    metrics=pd.DataFrame(collected)
    metrics.to_csv(OUT/'fold_comparison.csv',index=False)
    summary=metrics.groupby('model')[['rmse','mae','r2','tail_mae','logical_edges','continuous_terms']].agg(['mean','std'])
    summary.to_csv(OUT/'comparison.csv')
    paired=[]
    for family in ['all','pure','hinge_only']:
        values=metrics[metrics.model==f'exploration_{family}'].set_index('fold')
        for name in ['ridge','forest','rrl_optimized']:
            other=metrics[metrics.model==name].set_index('fold')
            for fold in range(5):
                paired.append(dict(family=family,reference=name,fold=fold,rmse_delta=values.loc[fold,'rmse']-other.loc[fold,'rmse']))
    pd.DataFrame(paired).to_csv(OUT/'paired_deltas.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,metric in zip(axes,['rmse','mae']):
        values=summary[metric]
        ax.bar(values.index,values['mean'],yerr=values['std'],capsize=3)
        ax.tick_params(axis='x',rotation=65)
        ax.set_title(f'{metric.upper()} (adaptive internal evaluation)')
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    checkpoint=torch.load(OUT/'final_model.pth',map_location='cpu',weights_only=False)
    inner=pd.concat([pd.read_csv(OUT/f'inner_{i}.csv') for i in range(5)])
    eligible=[i for i,c in enumerate(selected) if all_heads()[c['head']]['kind']!='hinge_only']
    cid=int(inner.groupby('candidate').rmse.mean().loc[eligible].idxmin())
    config=json.loads((OUT/'final_config.json').read_text())
    assert config['candidate']==cid
    assert config['network']==networks()[selected[cid]['network']]
    assert config['head']==all_heads()[selected[cid]['head']]
    if config['head']['kind']=='adam':
        assert checkpoint['head'] is None
    else:
        assert checkpoint['head']['kind']==config['head']['kind']
        assert checkpoint['head']['alpha']==config['head']['alpha']
    _,state=prepare(frame,transform=checkpoint['state']['transform'])
    for key in ['features','categories','mean','std','transform']:
        assert state[key]==checkpoint['state'][key]
    model=RRL(**checkpoint['rrl_args']);model.net.load_state_dict(checkpoint['model_state_dict'])
    graph=json.loads((OUT/'final_rules.json').read_text())
    err=float(np.max(np.abs(predict_refit(model,checkpoint['state'],frame,checkpoint['head'])-predict_graph(graph,frame))))
    assert err<1e-4
    audit=dict(folds=5,samples=len(frame),development_records=len(dev),candidates=len(selected),
               oof_max_rule_error=max_error,checkpoint_max_rule_error=err,
               caveat='Adaptive development reused data; no independent confirmation.')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(summary.round(5).to_string());print('AUDIT',audit)


if __name__=='__main__':
    torch.set_num_threads(1)
    main()

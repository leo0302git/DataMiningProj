"""Audit expanded baseline and RRL experiments without hiding stronger controls."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.eda import load_abalone
from experiments.abalone_round3 import OUT, networks, heads, rf_configs, smooth_configs, rf_predict, smooth
from experiments.run_abalone import scores
from models.rrl.abalone import prepare, RRL
from models.rrl.abalone_refine import predict_graph, predict_refit


def main():
    rf=pd.read_csv(OUT/'rf_development.csv')
    ranking=rf.groupby('candidate').agg(mean_rmse=('rmse','mean'),seed_std=('rmse','std')).sort_values('mean_rmse')
    ranking['config']=[json.dumps(rf_configs()[i]) for i in ranking.index]
    ranking.to_csv(OUT/'rf_ranking.csv')
    if not (OUT/'final_model.pth').exists():
        print(ranking.head(6).to_string());return
    frame=load_abalone();design=json.loads((OUT/'design.json').read_text())
    assert design['data_sha256']==hashlib.sha256((ROOT/'data/raw/abalone/abalone.data').read_bytes()).hexdigest()
    assert design['networks']==networks() and design['rf']==rf_configs() and design['heads']==heads()
    selected=json.loads((OUT/'shortlist.json').read_text());candidates=selected['rrl']
    all_rows=[];choices=[];rule_error=0.;rf_error=0.;matched=[]
    for family in ['rf','smooth','rrl']:
        indices=[]
        for fold,(a,b) in enumerate(KFold(5,shuffle=True,random_state=0).split(frame)):
            pred=pd.read_csv(OUT/f'{family}_fold_{fold}.csv')
            assert np.array_equal(pred['index'],b) and np.array_equal(pred.Rings,frame.iloc[b].Rings)
            indices.extend(pred['index'])
            row=json.loads((OUT/f'{family}_metrics_{fold}.json').read_text());cid=row['candidate']
            for key,value in scores(pred.Rings.to_numpy(),pred.prediction.to_numpy()).items():
                assert np.isclose(row[key],value)
            search=pd.read_csv(OUT/f'inner_{fold}.csv');part=search[search.family==family]
            assert not part.duplicated(['candidate','inner_fold']).any()
            expected=list(range(len(candidates))) if family=='rrl' else selected[family]
            assert sorted(part.candidate.unique())==sorted(expected) and len(part)==3*len(expected)
            assert cid==int(part.groupby('candidate').rmse.mean().idxmin())
            if family=='rf':
                actual=rf_predict(frame.iloc[a],frame.iloc[b],rf_configs()[cid],42+fold)
                rf_error=max(rf_error,float(np.max(np.abs(actual-pred.prediction))))
                assert rf_error<1e-8
                choices.append(dict(fold=fold,candidate=cid,**rf_configs()[cid],inner_rmse=float(part[part.candidate==cid].rmse.mean())))
            elif family=='smooth':
                assert np.allclose(smooth(frame.iloc[a],frame.iloc[b],smooth_configs()[cid]),pred.prediction,atol=1e-7)
            else:
                pack=json.loads((OUT/f'rrl_fold_{fold}.json').read_text())
                c=candidates[cid];assert len(pack['members'])==c['members']
                for g in pack['members']:
                    _,state=prepare(frame.iloc[a],transform=g['preprocessing']['transform'])
                    for key in ['features','categories','mean','std','transform']: assert state[key]==g['preprocessing'][key]
                    spec=heads()[c['head']]
                    assert g['head_alpha']==spec['alpha'] and g['continuous_alpha']==spec['continuous_alpha']
                actual=np.mean([predict_graph(g,frame.iloc[b]) for g in pack['members']],axis=0)
                error=float(np.max(np.abs(actual-pred.prediction)));rule_error=max(rule_error,error);assert error<1e-4
                matched.append(dict(fold=fold,rrl_rmse=row['rmse'],without_rules_rmse=row['matched_no_rules_rmse'],
                    delta=row['rmse']-row['matched_no_rules_rmse'],rule_contribution_std=row['rule_contribution_std']))
            all_rows.append(dict(model='round3_'+family,**row))
        assert sorted(indices)==list(range(len(frame)))
    for name in ['ridge','forest','rrl_optimized']:
        for i in range(5):
            r=json.loads((ROOT/f'results/abalone/{name}/fold_{i}_metrics.json').read_text())
            all_rows.append(dict(model=name,**{k:v for k,v in r.items() if k!='config'}))
    for name in ['all','hinge_only']:
        for i in range(5):
            r=json.loads((ROOT/f'results/abalone/exploration/{name}_metrics_{i}.json').read_text())
            all_rows.append(dict(model='round2_'+name,**r))
    rows=pd.DataFrame(all_rows);rows.to_csv(OUT/'fold_comparison.csv',index=False)
    summary=rows.groupby('model')[['rmse','mae','r2','tail_mae','logical_edges']].agg(['mean','std'])
    summary.to_csv(OUT/'comparison.csv');pd.DataFrame(choices).to_csv(OUT/'rf_selected.csv',index=False)
    pd.DataFrame(matched).to_csv(OUT/'matched_rule_ablation.csv',index=False)
    delta=[]
    rrl=rows[rows.model=='round3_rrl'].set_index('fold')
    for name in ['round3_rf','round3_smooth','forest','round2_all']:
        ref=rows[rows.model==name].set_index('fold')
        for i in range(5): delta.append(dict(reference=name,fold=i,rmse_delta=rrl.loc[i,'rmse']-ref.loc[i,'rmse']))
    pd.DataFrame(delta).to_csv(OUT/'paired_deltas.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,metric in zip(axes,['rmse','mae']):
        v=summary[metric];ax.bar(v.index,v['mean'],yerr=v['std'],capsize=3)
        ax.tick_params(axis='x',rotation=65);ax.set_title(metric.upper()+' (adaptive internal evaluation)')
    fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    searches=pd.concat([pd.read_csv(OUT/f'inner_{i}.csv') for i in range(5)])
    best=int(searches[searches.family=='rrl'].groupby('candidate').rmse.mean().idxmin())
    conf=json.loads((OUT/'final_config.json').read_text());c=candidates[best]
    assert conf['candidate']==best and conf['network']==networks()[c['network']] and conf['head']==heads()[c['head']]
    checkpoints=torch.load(OUT/'final_model.pth',map_location='cpu',weights_only=False)
    pack=json.loads((OUT/'final_rules.json').read_text());assert len(checkpoints)==len(pack['members'])==c['members']
    preds=[]
    for cp in checkpoints:
        model=RRL(**cp['rrl_args']);model.net.load_state_dict(cp['model_state_dict'])
        preds.append(predict_refit(model,cp['state'],frame,cp['head']))
    error=float(np.max(np.abs(np.mean(preds,axis=0)-np.mean([predict_graph(g,frame) for g in pack['members']],axis=0))))
    assert error<1e-4
    audit=dict(samples=len(frame),folds=5,rf_development_fits=len(rf),rrl_development_rows=len(pd.read_csv(OUT/'rrl_development.csv')),
        inner_score_rows=len(searches),oof_rule_error=rule_error,rf_retrain_prediction_error=rf_error,checkpoint_error=error,
        limitation='Adaptive development with previously observed data, not independent confirmation')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2));print(summary.round(6).to_string());print(audit)


if __name__=='__main__':
    torch.set_num_threads(1);main()

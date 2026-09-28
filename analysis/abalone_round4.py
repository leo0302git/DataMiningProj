"""Audit fourth-round selection, executable explanations, and paired comparisons."""
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
from experiments.abalone_round4 import OUT, networks, heads, select
from experiments.run_abalone import scores
from models.rrl.abalone import prepare, predict_rules, RRL
from models.rrl.abalone_refine import extra_features, gate_features, predict_graph, predict_refit
from models.rrl.abalone_refine import refit
from experiments.abalone_round3 import train_rrl
from experiments.explore_abalone import export


def solve(X,y,penalties):
    design=np.column_stack([np.ones(len(X)),X])
    coef=np.linalg.solve(design.T@design+len(X)*np.diag(np.r_[0,penalties]),design.T@y)
    return coef[0],coef[1:]


def check_graph(g,train,test,spec):
    state=g['preprocessing'];X,_=prepare(train,state);Xt,_=prepare(test,state)
    _,expected=prepare(train,transform=state['transform'])
    for key in ['features','categories','mean','std','transform']: assert state[key]==expected[key]
    R=predict_rules(g,train,return_rules=True).astype(float);Rt=predict_rules(g,test,return_rules=True).astype(float)
    B=extra_features(X,state,g['hinge_knots']);Bt=extra_features(Xt,state,g['hinge_knots'])
    y=train.Rings.to_numpy(float);base=np.column_stack([R,B]);baset=np.column_stack([Rt,Bt])
    penalties=np.r_[np.full(R.shape[1],spec['alpha']),np.full(B.shape[1],spec['continuous_alpha'])]
    bias,w=solve(base,y,penalties);plain=bias+baset@w
    residual=y-(bias+base@w)
    terms=g.get('gates',[])
    if spec.get('gate_count'):
        options=[]
        for r in np.flatnonzero((R.mean(0)>=.05)&(R.mean(0)<=.95)):
            for j in range(len(state['categories'])-1,X.shape[1]):
                center=float(X[R[:,r].astype(bool),j].astype(float).mean())
                col=R[:,r]*(X[:,j].astype(float)-center);scale=float(col.std())
                if scale>1e-8:
                    options.append((abs(float((col/scale)@residual)),dict(rule=int(r),feature=j,center=center,scale=scale)))
        chosen=[t for _,t in sorted(options,key=lambda pair:-pair[0])[:spec['gate_count']]]
        # Numerically identical duplicate gates can exchange rank in independent solvers.
        assert len(terms)==len(chosen)
        boundary=min([v for v,_ in sorted(options,key=lambda pair:-pair[0])[:len(chosen)]],default=0)
        for t in terms:
            active=R[:,t['rule']].astype(bool)
            assert .05<=active.mean()<=.95
            assert np.isclose(t['center'],X[active,t['feature']].astype(float).mean())
            raw=R[:,t['rule']]*(X[:,t['feature']].astype(float)-t['center'])
            assert np.isclose(t['scale'],raw.std())
            assert abs(float((raw/t['scale'])@residual))>=boundary-1e-7
        G=gate_features(X,R,terms);Gt=gate_features(Xt,Rt,terms)
        base=np.column_stack([base,G]);baset=np.column_stack([baset,Gt])
        penalties=np.r_[penalties,np.full(len(terms),spec['gate_alpha'])]
        assert g['gate_alpha']==spec['gate_alpha'] and g['gate_count']==spec['gate_count']
    else: assert not terms
    assert g['head_alpha']==spec['alpha'] and g['continuous_alpha']==spec['continuous_alpha']
    bias,w=solve(base,y,penalties)
    assert np.allclose(bias+baset@w,predict_graph(g,test),atol=1e-6)
    bias,w=solve(B,y,np.full(B.shape[1],spec['continuous_alpha']))
    return plain,bias+Bt@w


def main():
    frame=load_abalone();candidates=select(frame)
    design=json.loads((OUT/'design.json').read_text());prior=ROOT/'results/abalone/round3'
    assert design['data_sha256']==hashlib.sha256((ROOT/'data/raw/abalone/abalone.data').read_bytes()).hexdigest()
    assert design['networks']==networks() and design['heads']==heads()
    assert design['data_sha256']==json.loads((prior/'design.json').read_text())['data_sha256']
    anchor=candidates.index(dict(network=0,head=4,members=2))
    previous_candidates=json.loads((prior/'shortlist.json').read_text())['rrl']
    previous_anchor=previous_candidates.index(dict(network=14,head=4,members=2))
    rows=[];inner=[];coverage=[];error=0.;ablations=[]
    for fold,(a,b) in enumerate(KFold(5,shuffle=True,random_state=0).split(frame)):
        search=pd.read_csv(OUT/f'inner_{fold}.csv');inner.append(search)
        assert len(search)==3*len(candidates) and not search.duplicated(['candidate','inner_fold']).any()
        assert sorted(search.candidate.unique())==list(range(len(candidates))) and sorted(search.inner_fold.unique())==[0,1,2]
        historical=pd.read_csv(prior/f'inner_{fold}.csv')
        old=historical[(historical.family=='rrl')&(historical.candidate==previous_anchor)].sort_values('inner_fold')
        assert np.allclose(search[search.candidate==anchor].sort_values('inner_fold').rmse,old.rmse,atol=1e-10)
        row=json.loads((OUT/f'rrl_metrics_{fold}.json').read_text());cid=int(search.groupby('candidate').rmse.mean().idxmin())
        assert row['candidate']==cid
        p=pd.read_csv(OUT/f'rrl_fold_{fold}.csv');coverage.extend(p['index'])
        assert np.array_equal(p['index'],b) and np.array_equal(p.Rings,frame.iloc[b].Rings)
        for key,value in scores(p.Rings.to_numpy(),p.prediction.to_numpy()).items(): assert np.isclose(row[key],value)
        pack=json.loads((OUT/f'rrl_fold_{fold}.json').read_text());assert len(pack['members'])==candidates[cid]['members']
        error=max(error,float(np.max(np.abs(p.prediction-predict_graph(pack,frame.iloc[b])))));assert error<1e-4
        controls=[check_graph(g,frame.iloc[a],frame.iloc[b],heads()[candidates[cid]['head']]) for g in pack['members']]
        no_gates=scores(frame.iloc[b].Rings.to_numpy(),np.mean([p[0] for p in controls],axis=0))['rmse']
        no_rules=scores(frame.iloc[b].Rings.to_numpy(),np.mean([p[1] for p in controls],axis=0))['rmse']
        assert np.isclose(no_gates,row['matched_no_gates_rmse']) and np.isclose(no_rules,row['matched_no_rules_rmse'])
        ablations.append(dict(fold=fold,full=row['rmse'],without_gates=no_gates,without_rules=no_rules,
                              gate_delta=row['rmse']-no_gates,rule_delta=row['rmse']-no_rules))
        rows.append(dict(split_seed=0,family='round4_rrl',**row))
        for family in ['rrl','rf','smooth']:
            reference=json.loads((prior/f'{family}_metrics_{fold}.json').read_text())
            pred=pd.read_csv(prior/f'{family}_fold_{fold}.csv');assert np.array_equal(pred['index'],b)
            rows.append(dict(split_seed=0,family='round3_'+family,**reference))
        old=json.loads((ROOT/f'results/abalone/forest/fold_{fold}_metrics.json').read_text())
        rows.append(dict(split_seed=0,family='original_rf',**{k:v for k,v in old.items() if k!='config'}))
    assert sorted(coverage)==list(range(len(frame)))
    # Winning outer models may contain no gates: explicitly audit the rejected branch too.
    devtrain=frame.iloc[design['train_indices']];devval=frame.iloc[design['validation_indices']]
    ranking=pd.read_csv(OUT/'development_ranking.csv');chosen=ranking[ranking.gate_count>0].iloc[0]
    spec=heads()[int(chosen['head'])]
    model,state,_=train_rrl(devtrain,networks()[int(chosen.network)],314)
    head=refit(model,state,devtrain,**spec);graph=export(model,state,devtrain,head,OUT/'gated_development_audit')
    check_graph(graph,devtrain,devval,spec)
    assert np.allclose(predict_graph(graph,devval),predict_refit(model,state,devval,head),atol=1e-6)
    dev=pd.read_csv(OUT/'development.csv')
    row=dev[(dev.network==chosen.network)&(dev['head']==chosen['head'])&(dev.seed==314)].iloc[0]
    assert np.isclose(row.rmse,scores(devval.Rings.to_numpy(),predict_graph(graph,devval))['rmse'])
    searches=pd.concat(inner);cid=int(searches.groupby('candidate').rmse.mean().idxmin());c=candidates[cid]
    conf=json.loads((OUT/'final_config.json').read_text())
    assert conf['candidate']==cid and conf['head']==heads()[c['head']] and conf['network']==networks()[c['network']]
    cp=torch.load(OUT/'final_model.pth',map_location='cpu',weights_only=False)
    pack=json.loads((OUT/'final_rules.json').read_text());assert len(cp)==len(pack['members'])==c['members']
    same_final=pack==json.loads((prior/'final_rules.json').read_text())
    preds=[]
    for checkpoint in cp:
        model=RRL(**checkpoint['rrl_args']);model.net.load_state_dict(checkpoint['model_state_dict'])
        preds.append(predict_refit(model,checkpoint['state'],frame.drop(columns='Rings'),checkpoint['head']))
    checkpoint_error=float(np.max(np.abs(np.mean(preds,axis=0)-predict_graph(pack,frame.drop(columns='Rings')))));assert checkpoint_error<1e-4
    repeat=pd.read_csv(OUT/'repeated_metrics.csv');assert len(repeat)==10 and not repeat.duplicated(['split_seed','fold']).any()
    assert json.loads((OUT/'repeat_design.json').read_text())['config']==conf
    repeat_difference=0.
    for seed in [17,42]:
        for fold,(_,b) in enumerate(KFold(5,shuffle=True,random_state=seed).split(frame)):
            pred=pd.read_csv(OUT/f'repeat_{seed}_{fold}.csv')
            assert np.array_equal(pred['index'],b) and np.array_equal(pred.Rings,frame.iloc[b].Rings)
            old=pd.read_csv(prior/f'repeat_{seed}_rrl_{fold}.csv');assert np.array_equal(old['index'],pred['index'])
            repeat_difference=max(repeat_difference,float(np.max(np.abs(old.prediction-pred.prediction))))
            r=repeat[(repeat.split_seed==seed)&(repeat.fold==fold)].iloc[0].to_dict()
            for key,value in scores(pred.Rings.to_numpy(),pred.prediction.to_numpy()).items(): assert np.isclose(r[key],value)
            rows.append(dict(family='round4_rrl',**r))
    previous=pd.read_csv(prior/'repeated_metrics.csv')
    if same_final: assert repeat_difference<1e-8
    for r in previous.to_dict('records'): r['family']='round3_'+r['family'];rows.append(r)
    data=pd.DataFrame(rows);data.to_csv(OUT/'fold_comparison.csv',index=False)
    summary=data.groupby(['split_seed','family'])[['rmse','mae','r2','tail_mae']].agg(['mean','std'])
    summary.to_csv(OUT/'comparison.csv');pd.DataFrame(ablations).to_csv(OUT/'matched_ablation.csv',index=False)
    deltas=data.pivot(index=['split_seed','fold'],columns='family',values='rmse')
    for ref in ['round3_rrl','round3_rf','round3_smooth']: deltas['delta_vs_'+ref]=deltas.round4_rrl-deltas[ref]
    deltas.to_csv(OUT/'paired_deltas.csv')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    labels=['RF (expanded)','RRL round 3','RRL round 4','No rules']
    for seed,ax in zip([0,17,42],axes):
        part=summary.loc[seed].loc[['round3_rf','round3_rrl','round4_rrl','round3_smooth']]['rmse']
        ax.errorbar(np.arange(len(labels)),part['mean'],yerr=part['std'],fmt='o',capsize=3)
        ax.set_xticks(np.arange(len(labels)),labels,rotation=50);ax.set_ylim(1.9,2.35)
        ax.set_title(f'Split seed {seed}');ax.set_ylabel('RMSE')
    fig.suptitle('Adaptive internal evaluation; mean +/- fold SD (not confidence intervals)')
    fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    audit=dict(samples=len(frame),development_rows=28*len(heads()),inner_rows=len(searches),repeated_rows=len(repeat),
               rule_error=error,checkpoint_error=checkpoint_error,training_only_gate_selection_verified=True,
               final_graph_identical_to_round3=same_final,repeat_prediction_difference_from_round3=repeat_difference,
               limitation='Adaptive reused data; not independent or statistically significant evidence')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(summary.round(6).to_string());print('AUDIT',audit);print('FINAL',conf)


if __name__=='__main__':
    torch.set_num_threads(1);main()

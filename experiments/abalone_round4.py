"""Frozen fourth-round search: local rule slopes, regularization, and averaging."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.eda import load_abalone
from experiments.abalone_round3 import train_rrl, networks as previous_networks
from experiments.explore_abalone import frozen, write_json, export
from experiments.run_abalone import scores
from models.rrl.abalone_refine import refit, predict_refit, predict_graph

OUT=ROOT/'results/abalone/round4'
SEEDS=[314,2718,10314,12718,20314,22718,30314,32718]


def networks():
    base=previous_networks()[14]
    return [base,previous_networks()[4],dict(base,structure='10@128'),dict(base,structure='40@128'),
            dict(base,structure='20@256'),dict(base,huber_delta=.5)]


def heads():
    plain=[dict(kind='hinge_rules',alpha=a,continuous_alpha=b)
           for a in [.03,.1,.3] for b in [.003,.01,.03]]
    gated=[dict(kind='hinge_rules',alpha=.1,continuous_alpha=b,gate_count=k,gate_alpha=g)
           for k in [32,96] for b in [.003,.01,.03] for g in [.03,.3]]
    return plain+gated


def design(frame):
    prior=json.loads((ROOT/'results/abalone/round3/design.json').read_text())
    frozen(OUT/'design.json',dict(networks=networks(),heads=heads(),seeds=SEEDS,
        members_by_network=[8,4,4,4,4,4],train_indices=prior['train_indices'],validation_indices=prior['validation_indices'],
        data_sha256=hashlib.sha256((ROOT/'data/raw/abalone/abalone.data').read_bytes()).hexdigest(),
        selection='For each gate count 0/32/96 keep top two distinct (network,members), plus prior anchor and best eight-member candidate',
        evaluation='Original five outer/three inner folds; choose by inner RMSE; compare frozen round3 RF and RRL; repeat fixed winner with split seeds 17/42',
        limitation='Adaptive development on previously observed data, not independent confirmation; no outer-driven additions'))
    return frame.iloc[prior['train_indices']],frame.iloc[prior['validation_indices']]


def develop(frame):
    train,val=design(frame);path=OUT/'development.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    for nid,config in enumerate(networks()):
        for seed in SEEDS[:8 if nid==0 else 4]:
            if any(r['network']==nid and r['seed']==seed for r in rows):
                assert sum(r['network']==nid and r['seed']==seed for r in rows)==len(heads())
                continue
            model,state,_=train_rrl(train,config,seed);predictions=[]
            for hid,spec in enumerate(heads()):
                head=refit(model,state,train,**spec);p=predict_refit(model,state,val,head)
                predictions.append(p)
                rows.append(dict(network=nid,head=hid,seed=seed,epochs=state['selected_epochs'],
                                 gate_terms=len(head.get('gates',[])),**scores(val.Rings.to_numpy(),p)))
            np.savez_compressed(OUT/f'dev_{nid}_{seed}.npz',prediction=np.stack(predictions))
            pd.DataFrame(rows).to_csv(path,index=False)
            print('DEV',nid,seed,'epoch',state['selected_epochs'],'best',min(r['rmse'] for r in rows[-len(heads()):]),flush=True)


def select(frame):
    _,val=design(frame);dev=pd.read_csv(OUT/'development.csv')
    assert len(dev)==28*len(heads()) and not dev.duplicated(['network','head','seed']).any()
    ranking=[]
    for nid in range(len(networks())):
        for members in ([2,4,8] if nid==0 else [2,4]):
            prediction=np.mean([np.load(OUT/f'dev_{nid}_{s}.npz')['prediction'] for s in SEEDS[:members]],axis=0)
            for hid,h in enumerate(heads()):
                ranking.append(dict(network=nid,head=hid,members=members,gate_count=h.get('gate_count',0),
                                    **scores(val.Rings.to_numpy(),prediction[hid])))
    ranked=pd.DataFrame(ranking).sort_values(['rmse','network','head','members'])
    ranked.to_csv(OUT/'development_ranking.csv',index=False)
    choices=[]
    for count in [0,32,96]:
        rows=ranked[ranked.gate_count==count].drop_duplicates(['network','members']).head(2)
        choices.extend(dict(network=int(r.network),head=int(r.head),members=int(r.members)) for r in rows.itertuples())
    anchor=dict(network=0,head=4,members=2)
    best8=ranked[ranked.members==8].iloc[0]
    for c in [anchor,dict(network=int(best8.network),head=int(best8['head']),members=8)]:
        if c not in choices: choices.append(c)
    frozen(OUT/'shortlist.json',choices)
    return choices


def fit_candidate(train,test,c,seed,prefix=None):
    predictions=[];plain=[];smooth=[];graphs=[];checkpoints=[];epochs=[]
    spec=heads()[c['head']]
    for m in range(c['members']):
        model,state,args=train_rrl(train,networks()[c['network']],seed+m*10000)
        head=refit(model,state,train,**spec)
        predictions.append(predict_refit(model,state,test,head));epochs.append(state['selected_epochs'])
        control=refit(model,state,train,'hinge_only',spec['continuous_alpha'])
        smooth.append(predict_refit(model,state,test,control))
        no_gates=refit(model,state,train,**{k:v for k,v in spec.items() if not k.startswith('gate_')})
        plain.append(predict_refit(model,state,test,no_gates))
        if prefix is not None:
            graphs.append(export(model,state,train,head,Path(str(prefix)+f'_member_{m}')))
            checkpoints.append(dict(model_state_dict=model.net.state_dict(),state=state,rrl_args=args,head=head))
    p=np.mean(predictions,axis=0)
    metadata=dict(members=c['members'],epochs=epochs,matched_no_rules_rmse=scores(test.Rings.to_numpy(),np.mean(smooth,axis=0))['rmse'],
                  matched_no_gates_rmse=scores(test.Rings.to_numpy(),np.mean(plain,axis=0))['rmse'])
    if prefix is not None:
        pack=dict(members=graphs,aggregation='arithmetic_mean')
        error=float(np.max(np.abs(p-predict_graph(pack,test))));assert error<1e-4
        write_json(prefix.with_suffix('.json'),pack)
        metadata.update(export_error=error,gate_terms=sum(len(g.get('gates',[])) for g in graphs),
            logical_edges=int(sum(np.sum(l['conjunction'])+np.sum(l['disjunction']) for g in graphs for l in g['layers'])),
            continuous_terms=sum(len(g['linear_weights']) for g in graphs))
    return p,metadata,checkpoints


def evaluate(frame,only_fold=None):
    candidates=select(frame)
    for fold,(a,b) in enumerate(KFold(5,shuffle=True,random_state=0).split(frame)):
        if only_fold is not None and fold!=only_fold: continue
        train,test=frame.iloc[a],frame.iloc[b];path=OUT/f'inner_{fold}.csv'
        rows=pd.read_csv(path).to_dict('records') if path.exists() else []
        for k,(tr,va) in enumerate(KFold(3,shuffle=True,random_state=100+fold).split(train)):
            proper,val=train.iloc[tr],train.iloc[va]
            for nid in sorted({c['network'] for c in candidates}):
                ids=[i for i,c in enumerate(candidates) if c['network']==nid]
                done={r['candidate'] for r in rows if r['inner_fold']==k}
                if all(i in done for i in ids): continue
                predictions={i:[] for i in ids}
                for m in range(max(candidates[i]['members'] for i in ids)):
                    model,state,_=train_rrl(proper,networks()[nid],1000+fold*10+k+m*10000)
                    for cid in ids:
                        if m<candidates[cid]['members']:
                            head=refit(model,state,proper,**heads()[candidates[cid]['head']])
                            predictions[cid].append(predict_refit(model,state,val,head))
                for cid in ids:
                    if cid not in done:
                        rows.append(dict(candidate=cid,inner_fold=k,**scores(val.Rings.to_numpy(),np.mean(predictions[cid],axis=0))))
                pd.DataFrame(rows).to_csv(path,index=False)
                print('INNER',fold,k,nid,flush=True)
        prefix=OUT/f'rrl_fold_{fold}'
        if prefix.with_suffix('.csv').exists(): continue
        cid=int(pd.DataFrame(rows).groupby('candidate').rmse.mean().idxmin())
        p,metadata,_=fit_candidate(train,test,candidates[cid],42+fold,prefix)
        pd.DataFrame(dict(index=test.index,Rings=test.Rings,prediction=p)).to_csv(prefix.with_suffix('.csv'),index=False)
        write_json(OUT/f'rrl_metrics_{fold}.json',dict(fold=fold,candidate=cid,**metadata,**scores(test.Rings.to_numpy(),p)))
        print('OUTER',fold,cid,scores(test.Rings.to_numpy(),p)['rmse'],flush=True)


def final(frame):
    candidates=select(frame)
    inner=pd.concat([pd.read_csv(OUT/f'inner_{i}.csv') for i in range(5)])
    cid=int(inner.groupby('candidate').rmse.mean().idxmin());c=candidates[cid]
    _,metadata,checkpoints=fit_candidate(frame,frame,c,2026,OUT/'final_rules')
    torch.save(checkpoints,OUT/'final_model.pth')
    write_json(OUT/'final_config.json',dict(candidate=cid,network=networks()[c['network']],head=heads()[c['head']],members=c['members'],
                                         epochs=metadata['epochs'],selection='mean inner RMSE, never outer scores'))


def repeat(frame):
    c=json.loads((OUT/'final_config.json').read_text());candidate=json.loads((OUT/'shortlist.json').read_text())[c['candidate']]
    frozen(OUT/'repeat_design.json',dict(split_seeds=[17,42],config=c,no_retuning=True,limitation='Split sensitivity, not independent data'))
    path=OUT/'repeated_metrics.csv';rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    for split_seed in [17,42]:
        for fold,(a,b) in enumerate(KFold(5,shuffle=True,random_state=split_seed).split(frame)):
            if any(r['split_seed']==split_seed and r['fold']==fold for r in rows): continue
            train,test=frame.iloc[a],frame.iloc[b]
            p,meta,_=fit_candidate(train,test,candidate,42+fold+split_seed*100)
            pd.DataFrame(dict(index=test.index,Rings=test.Rings,prediction=p)).to_csv(OUT/f'repeat_{split_seed}_{fold}.csv',index=False)
            rows.append(dict(split_seed=split_seed,fold=fold,**meta,**scores(test.Rings.to_numpy(),p)))
            pd.DataFrame(rows).to_csv(path,index=False)
            print('REPEAT',split_seed,fold,scores(test.Rings.to_numpy(),p)['rmse'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['design','develop','select','evaluate','final','repeat'])
    parser.add_argument('--fold',type=int,choices=range(5));args=parser.parse_args()
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='evaluate': evaluate(load_abalone(),args.fold)
    else: globals()[args.stage](load_abalone())

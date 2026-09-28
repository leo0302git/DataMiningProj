"""Expanded RF search versus residual/early-stopped/group-regularized RRL."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.eda import load_abalone
from experiments.run_abalone import scores
from experiments.explore_abalone import frozen, write_json, export
from models.manual_glm.ridge import RidgeRegression
from models.rrl.abalone import prepare, fit
from models.rrl.abalone_refine import extra_features, refit, predict_refit, predict_graph

OUT=ROOT/'results/abalone/round3'
SEEDS=[314,2718]


def networks():
    base=dict(threshold='quantile',structure='20@64',lr=.002,wd=0.,alpha=.999,beta=8,gamma=3,
              epochs=160,loss='huber',transform='log',use_not=False,residual=False)
    configs=[dict(base, residual=r, loss=loss, epochs=e) for r in [False,True]
             for loss in ['mse','huber'] for e in [20,80,160]]
    for r in [False,True]:
        for structure in ['10@32','20@64','20@128']:
            configs.append(dict(base,residual=r,structure=structure,epochs=320,stop_patience=3,monitor_every=10))
        for penalty in [1e-5,1e-4,1e-3]:
            configs.append(dict(base,residual=r,edge_lambda=penalty))
        for change in [dict(structure='40@64'),dict(structure='20@32'),dict(transform='standard'),
                       dict(use_not=True),dict(gamma=1),dict(lr=.005)]:
            configs.append(dict(base,residual=r,**change))
    configs.append(dict(base,epochs=320))
    return configs


def heads():
    return [dict(kind='hinge_rules',alpha=a,continuous_alpha=b)
            for a in [.01,.1,1.,10.] for b in [.001,.01,.1]]


def rf_configs():
    configs=[dict(n_estimators=150,max_depth=d,min_samples_leaf=l,max_features=1.,max_samples=None)
             for d in [8,None] for l in [1,5,10]]
    rng=np.random.default_rng(20260929)
    while len(configs)<72:
        c=dict(n_estimators=int(rng.choice([150,400,800])),max_depth=[6,8,12,20,None][int(rng.integers(5))],
            min_samples_leaf=int(rng.choice([1,2,4,8,10,16,24])),max_features=float(rng.choice([.4,.7,1.])),
            max_samples=[.6,.8,None][int(rng.integers(3))])
        if c not in configs: configs.append(c)
    return configs


def smooth_configs():
    return [dict(count=k,transform=t,alpha=a) for k in [10,20,40] for t in ['standard','log'] for a in [.001,.01,.1]]


def smooth(train,test,config):
    X,state=prepare(train,transform=config['transform']);Xt,_=prepare(test,state)
    k=config['count']; nd=len(state['categories'])-1
    knots=np.quantile(X[:,nd:],np.arange(1,k+1)/(k+1),axis=0).astype('float32').tolist()
    model=RidgeRegression(config['alpha']).fit(extra_features(X,state,knots),train.Rings.to_numpy(float))
    return model.predict(extra_features(Xt,state,knots))


def targets(train,valid,config):
    if not config['residual']: return train,valid
    spec=dict(count=20,transform=config['transform'],alpha=.01)
    a=train.copy();a['Rings']=train.Rings.to_numpy()-smooth(train,train,spec)
    if valid is None: return a,None
    b=valid.copy();b['Rings']=valid.Rings.to_numpy()-smooth(train,valid,spec)
    return a,b


def train_rrl(train,config,seed):
    config=dict(config); selected=config['epochs']
    if config.get('stop_patience'):
        a,b=train_test_split(np.arange(len(train)),test_size=.2,random_state=seed+900)
        proper,stop=targets(train.iloc[a],train.iloc[b],config)
        _,state,_=fit(proper,config,seed,validation=stop)
        selected=state['best_epoch']
        config.pop('stop_patience')
        config['epochs']=selected
    used,_=targets(train,None,config)
    model,state,args=fit(used,config,seed)
    state['selected_epochs']=selected
    return model,state,args


def rf_predict(train,test,config,seed):
    X,state=prepare(train);Xt,_=prepare(test,state)
    return RandomForestRegressor(**config,random_state=seed,n_jobs=2).fit(X,train.Rings).predict(Xt)


def design(frame):
    prior=json.loads((ROOT/'results/abalone/exploration/design.json').read_text())
    frozen(OUT/'design.json',dict(networks=networks(),heads=heads(),rf=rf_configs(),smooth=smooth_configs(),seeds=SEEDS,
        train_indices=prior['train_indices'],validation_indices=prior['validation_indices'],
        data_sha256=hashlib.sha256((ROOT/'data/raw/abalone/abalone.data').read_bytes()).hexdigest(),
        protocol='Adaptive development; top two RRL per residual flag and single/ensemble mode, RF top six, smooth top four; three inner/five outer folds',
        limitation='Previously observed data; not independent confirmation; no outer-driven additions in this round'))
    return frame.iloc[prior['train_indices']],frame.iloc[prior['validation_indices']]


def rf_develop(frame):
    train,val=design(frame);path=OUT/'rf_development.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    done={(int(r['candidate']),int(r['seed'])) for r in rows}
    for cid,c in enumerate(rf_configs()):
        for seed in SEEDS:
            if (cid,seed) in done: continue
            p=rf_predict(train,val,c,seed)
            rows.append(dict(candidate=cid,seed=seed,**scores(val.Rings.to_numpy(),p)))
            pd.DataFrame(rows).to_csv(path,index=False)
        print('RF DEV',cid,flush=True)
    if not (OUT/'smooth_development.csv').exists():
        pd.DataFrame([dict(candidate=i,**scores(val.Rings.to_numpy(),smooth(train,val,c)))
                      for i,c in enumerate(smooth_configs())]).to_csv(OUT/'smooth_development.csv',index=False)


def rrl_develop(frame):
    train,val=design(frame);path=OUT/'rrl_development.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    done={(int(r['network']),int(r['seed'])) for r in rows}
    for nid,c in enumerate(networks()):
        for seed in SEEDS:
            if (nid,seed) in done:
                assert sum(r['network']==nid and r['seed']==seed for r in rows)==len(heads())
                continue
            model,state,_=train_rrl(train,c,seed);predictions=[]
            for hid,spec in enumerate(heads()):
                head=refit(model,state,train,**spec);p=predict_refit(model,state,val,head)
                predictions.append(p)
                rows.append(dict(network=nid,head=hid,seed=seed,selected_epochs=state['selected_epochs'],
                                 **scores(val.Rings.to_numpy(),p)))
            np.savez_compressed(OUT/f'dev_{nid}_{seed}.npz',prediction=np.stack(predictions))
            pd.DataFrame(rows).to_csv(path,index=False)
            print('RRL DEV',nid,seed,'epochs',state['selected_epochs'],'best',min(r['rmse'] for r in rows[-len(heads()):]),flush=True)


def select(frame):
    _,val=design(frame)
    dev=pd.read_csv(OUT/'rrl_development.csv');assert len(dev)==len(networks())*2*len(heads())
    rf=pd.read_csv(OUT/'rf_development.csv');assert len(rf)==len(rf_configs())*2
    ranking=[]
    for nid,c in enumerate(networks()):
        ensemble=np.mean([np.load(OUT/f'dev_{nid}_{seed}.npz')['prediction'] for seed in SEEDS],axis=0)
        for hid in range(len(heads())):
            ranking.append(dict(network=nid,head=hid,members=1,residual=c['residual'],
                rmse=float(dev[(dev.network==nid)&(dev['head']==hid)].rmse.mean())))
            ranking.append(dict(network=nid,head=hid,members=2,residual=c['residual'],
                rmse=scores(val.Rings.to_numpy(),ensemble[hid])['rmse']))
    ranked=pd.DataFrame(ranking).sort_values(['rmse','network','head'])
    ranked.to_csv(OUT/'rrl_ranking.csv',index=False)
    shortlisted=ranked.groupby(['residual','members'],sort=False).head(2)
    rrl=[dict(network=int(r.network),head=int(r.head),members=int(r.members)) for r in shortlisted.itertuples()]
    rf_ids=rf.groupby('candidate').rmse.mean().sort_values().head(6).index.astype(int).tolist()
    sm_ids=pd.read_csv(OUT/'smooth_development.csv').sort_values('rmse').head(4).candidate.astype(int).tolist()
    frozen(OUT/'shortlist.json',dict(rrl=rrl,rf=rf_ids,smooth=sm_ids))
    return rrl,rf_ids,sm_ids


def evaluate(frame):
    candidates,rf_ids,sm_ids=select(frame)
    for fold,(a,b) in enumerate(KFold(5,shuffle=True,random_state=0).split(frame)):
        train,test=frame.iloc[a],frame.iloc[b];path=OUT/f'inner_{fold}.csv'
        rows=pd.read_csv(path).to_dict('records') if path.exists() else []
        for k,(tr,va) in enumerate(KFold(3,shuffle=True,random_state=100+fold).split(train)):
            proper,val=train.iloc[tr],train.iloc[va]
            for family,ids,func,pool in [('rf',rf_ids,rf_predict,rf_configs()),('smooth',sm_ids,smooth,smooth_configs())]:
                for cid in ids:
                    if any(r['family']==family and r['candidate']==cid and r['inner_fold']==k for r in rows): continue
                    p=func(proper,val,pool[cid],1000+fold*10+k) if family=='rf' else func(proper,val,pool[cid])
                    rows.append(dict(family=family,candidate=cid,inner_fold=k,**scores(val.Rings.to_numpy(),p)))
                    pd.DataFrame(rows).to_csv(path,index=False)
            for nid in sorted({c['network'] for c in candidates}):
                ids=[i for i,c in enumerate(candidates) if c['network']==nid]
                done={r['candidate'] for r in rows if r['family']=='rrl' and r['inner_fold']==k}
                if all(i in done for i in ids): continue
                max_members=max(candidates[i]['members'] for i in ids);predictions={i:[] for i in ids}
                for m in range(max_members):
                    model,state,_=train_rrl(proper,networks()[nid],1000+fold*10+k+m*10000)
                    for cid in ids:
                        if m<candidates[cid]['members']:
                            head=refit(model,state,proper,**heads()[candidates[cid]['head']])
                            predictions[cid].append(predict_refit(model,state,val,head))
                for cid in ids:
                    if cid not in done:
                        rows.append(dict(family='rrl',candidate=cid,inner_fold=k,
                            **scores(val.Rings.to_numpy(),np.mean(predictions[cid],axis=0))))
                pd.DataFrame(rows).to_csv(path,index=False)
                print('INNER',fold,k,nid,flush=True)
        data=pd.DataFrame(rows)
        for family in ['rf','smooth','rrl']:
            prefix=OUT/f'{family}_fold_{fold}'
            if prefix.with_suffix('.csv').exists(): continue
            cid=int(data[data.family==family].groupby('candidate').rmse.mean().idxmin())
            metadata=dict(fold=fold,candidate=cid)
            if family=='rf': p=rf_predict(train,test,rf_configs()[cid],42+fold)
            elif family=='smooth': p=smooth(train,test,smooth_configs()[cid])
            else:
                c=candidates[cid];predictions=[];graphs=[];epochs=[]
                for m in range(c['members']):
                    model,state,_=train_rrl(train,networks()[c['network']],42+fold+m*10000)
                    head=refit(model,state,train,**heads()[c['head']])
                    predictions.append(predict_refit(model,state,test,head));epochs.append(state['selected_epochs'])
                    graphs.append(export(model,state,train,head,OUT/f'rrl_fold_{fold}_member_{m}'))
                p=np.mean(predictions,axis=0)
                error=float(np.max(np.abs(p-np.mean([predict_graph(g,test) for g in graphs],axis=0))))
                assert error<1e-4
                write_json(prefix.with_suffix('.json'),dict(members=graphs,aggregation='arithmetic_mean'))
                metadata.update(export_error=error,epochs=epochs,members=c['members'],
                    logical_edges=int(sum(np.sum(l['conjunction'])+np.sum(l['disjunction']) for g in graphs for l in g['layers'])))
            pd.DataFrame(dict(index=test.index,Rings=test.Rings,prediction=p)).to_csv(prefix.with_suffix('.csv'),index=False)
            write_json(OUT/f'{family}_metrics_{fold}.json',dict(**metadata,**scores(test.Rings.to_numpy(),p)))
            print('OUTER',family,fold,scores(test.Rings.to_numpy(),p)['rmse'],flush=True)


def final(frame):
    candidates,_,_=select(frame)
    data=pd.concat([pd.read_csv(OUT/f'inner_{i}.csv') for i in range(5)])
    cid=int(data[data.family=='rrl'].groupby('candidate').rmse.mean().idxmin());c=candidates[cid]
    graphs=[];checkpoints=[]
    for m in range(c['members']):
        model,state,args=train_rrl(frame,networks()[c['network']],2026+m*10000)
        head=refit(model,state,frame,**heads()[c['head']])
        graphs.append(export(model,state,frame,head,OUT/f'final_member_{m}'))
        checkpoints.append(dict(model_state_dict=model.net.state_dict(),state=state,rrl_args=args,head=head))
    write_json(OUT/'final_rules.json',dict(members=graphs,aggregation='arithmetic_mean'))
    torch.save(checkpoints,OUT/'final_model.pth')
    write_json(OUT/'final_config.json',dict(candidate=cid,network=networks()[c['network']],head=heads()[c['head']],members=c['members']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['design','rf_develop','rrl_develop','select','evaluate','final'])
    args=parser.parse_args();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    globals()[args.stage](load_abalone())

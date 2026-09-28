"""Adaptive second-round development; never overwrite first-round evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.eda import load_abalone
from experiments.run_abalone import scores, baseline
from models.rrl.abalone import fit, export_rules
from models.rrl.abalone_refine import representation, refit, predict_refit, graph_with_head, predict_graph

OUT = ROOT/'results/abalone/exploration'


def networks():
    base = dict(threshold='quantile',structure='10@32',lr=.002,wd=0.,alpha=.999,beta=8,gamma=3,
                epochs=80,loss='huber',transform='log',use_not=False)
    changes = [{}, dict(loss='mse'), dict(epochs=160), dict(epochs=320), dict(epochs=640),
        dict(loss='mse',epochs=320), dict(loss='huber',huber_delta=.5,epochs=320),
        dict(loss='huber',huber_delta=2.,epochs=320), dict(structure='10@64',epochs=320),
        dict(structure='20@64',epochs=320), dict(structure='20@128',epochs=320),
        dict(structure='40@64',epochs=320), dict(structure='10@64@32',epochs=320),
        dict(structure='10@64',epochs=320,loss='mse'), dict(structure='20@64',epochs=320,loss='mse'),
        dict(structure='10@64',epochs=320,lr=.005), dict(structure='10@64',epochs=320,lr=.001),
        dict(structure='10@64',epochs=320,decay=.5), dict(structure='10@64',epochs=320,decay=1.),
        dict(structure='10@64',epochs=320,batch_size=64), dict(structure='10@64',epochs=320,batch_size=256),
        dict(structure='10@64',epochs=320,use_not=True), dict(structure='10@64',epochs=320,transform='standard'),
        dict(structure='10@64',epochs=320,threshold='random'),
        dict(structure='10@64',epochs=320,threshold='supervised'), dict(structure='10@64',epochs=320,gamma=1),
        dict(structure='10@64',epochs=320,alpha=.9,beta=3), dict(structure='10@64',epochs=320,wd=.0001)]
    return [dict(base,**change) for change in changes]


def heads():
    return [dict(kind='adam',alpha=0.)] + [dict(kind=kind,alpha=a)
        for kind in ['rules','linear_rules'] for a in [1e-5,.0001,.001,.01,.1]]


def all_heads():
    return heads()+[dict(kind=kind,alpha=a) for kind in ['hinge_rules','hinge_only'] for a in [.0001,.001,.01,.1,1.]]


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2))


def frozen(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Design mismatch: {path}'
    else:
        write_json(path,value)


def get_head(model,state,train,spec):
    return None if spec['kind']=='adam' else refit(model,state,train,**spec)


def export(model,state,train,head,path):
    graph=graph_with_head(export_rules(model,state,train,path),head)
    write_json(path.with_suffix('.json'),graph)
    if head is not None:
        text=path.with_suffix('.md').read_text().split('\nRings =')[0]
        text += f'\n\nRings = {head["bias"]:.10g} + sum(w_j * final_rule_j) + sum(v_k * X_k)\n'
        text += '\n固定规则后在训练集重拟合输出层；X顺序为非基准Sex独热列、预处理后的连续输入。log版先log1p再标准化。JSON为全精度可执行解释。\n'
        if head.get('hinge_knots') is not None:
            text += '\n连续项还包括 max(X_j-c_jk,0)，按特征再按阈值排列。其阈值采用JSON的hinge_knots标准化坐标，随后列出对应权重。\n'
        text += '\n规则权重：'+json.dumps(head['weights'])+'\n线性直连权重：'+json.dumps(head['linear_weights'])+'\n'
        if head.get('gates'):
            text += '\n上式还需加 sum(q_k * R_rule * (X_feature-center)/scale)。仅当该硬规则成立时贡献局部斜率；X使用上述预处理坐标，rule和feature为零起始下标。\n'
            text += '\n局部斜率项：'+json.dumps(head['gates'])+'\n对应权重：'+json.dumps(head['gate_weights'])+'\n'
        path.with_suffix('.md').write_text(text)
    return graph


def develop(frame):
    outer_train=list(KFold(5,shuffle=True,random_state=0).split(frame))[0][0]
    tr,va=train_test_split(outer_train,test_size=.2,random_state=2718)
    frozen(OUT/'design.json',dict(networks=networks(),heads=heads(),seeds=[314,2718],
        train_indices=tr.tolist(),validation_indices=va.tolist(),
        data_sha256=hashlib.sha256((ROOT/'data/raw/abalone/abalone.data').read_bytes()).hexdigest(),
        selection='top two per head family by two-seed mean development RMSE; fixed original winner anchor',
        limitation='adaptive internal evaluation, not independent confirmation'))
    path=OUT/'development.csv'
    records=pd.read_csv(path).to_dict('records') if path.exists() else []
    done={(int(r['network']),int(r['seed'])) for r in records}
    for nid,config in enumerate(networks()):
        for seed in [314,2718]:
            if (nid,seed) in done:
                assert sum(r['network']==nid and r['seed']==seed for r in records)==len(heads())
                continue
            history=[]
            start=time.monotonic()
            model,state,_=fit(frame.iloc[tr],config,seed,frame.iloc[va],history)
            write_json(OUT/f'curve_{nid}_{seed}.json',history)
            _,rules=representation(model,state,frame.iloc[tr])
            diagnostics=dict(active_rules=int(((rules.mean(0)>0)&(rules.mean(0)<1)).sum()),
                             distinct_rules=int(np.unique(rules,axis=1).shape[1]))
            for hid,spec in enumerate(heads()):
                head=get_head(model,state,frame.iloc[tr],spec)
                prediction=predict_refit(model,state,frame.iloc[va],head)
                train_pred=predict_refit(model,state,frame.iloc[tr],head)
                records.append(dict(network=nid,head=hid,kind=spec['kind'],alpha=spec['alpha'],seed=seed,
                    **scores(frame.iloc[va].Rings.to_numpy(),prediction),
                    train_rmse=scores(frame.iloc[tr].Rings.to_numpy(),train_pred)['rmse'],**diagnostics,
                    seconds=time.monotonic()-start))
            pd.DataFrame(records).to_csv(path,index=False)
            print('DEV',nid,seed,'best',min(r['rmse'] for r in records[-len(heads()):]),
                  'seconds',round(time.monotonic()-start),flush=True)
    if not (OUT/'development_baselines.json').exists():
        rows=[]
        for kind,pool in [('ridge',[dict(alpha=a) for a in [0,.001,.01,.1,1]]),
                          ('forest',[dict(max_depth=d,min_samples_leaf=l) for d in [8,None] for l in [1,5,10]])]:
            for config in pool:
                rows.append(dict(model=kind,config=config,**scores(frame.iloc[va].Rings.to_numpy(),
                    baseline(kind,config,frame.iloc[tr],frame.iloc[va]))))
        write_json(OUT/'development_baselines.json',rows)


def extend(frame):
    dev=pd.read_csv(OUT/'development.csv')
    assert len(dev)==len(networks())*2*len(heads())
    best=dev[dev.kind=='linear_rules'].groupby(['network','head']).rmse.mean().sort_values().reset_index()
    ids=sorted(set([0]+best.drop_duplicates('network').head(2).network.astype(int).tolist()))
    frozen(OUT/'extension_design.json',dict(network_ids=ids,heads=all_heads()[len(heads()):],
        reason='Observed benefit of continuous skip motivates piecewise-linear terms; additive-only control included',
        status='adaptive extension after initial development, before extension results'))
    design=json.loads((OUT/'design.json').read_text())
    train=frame.iloc[design['train_indices']];val=frame.iloc[design['validation_indices']]
    path=OUT/'extension.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    for nid in ids:
        for seed in [314,2718]:
            if any(r['network']==nid and r['seed']==seed for r in rows):
                assert sum(r['network']==nid and r['seed']==seed for r in rows)==10
                continue
            model,state,_=fit(train,networks()[nid],seed)
            for hid in range(len(heads()),len(all_heads())):
                spec=all_heads()[hid];head=get_head(model,state,train,spec)
                pred=predict_refit(model,state,val,head)
                rows.append(dict(network=nid,head=hid,seed=seed,kind=spec['kind'],alpha=spec['alpha'],
                    **scores(val.Rings.to_numpy(),pred),train_rmse=scores(train.Rings.to_numpy(),predict_refit(model,state,train,head))['rmse']))
            pd.DataFrame(rows).to_csv(path,index=False)
            print('EXTENSION',nid,seed,'best',min(r['rmse'] for r in rows[-10:]),flush=True)


def shortlist():
    rows=pd.read_csv(OUT/'development.csv')
    assert len(rows)==len(networks())*2*len(heads())
    extra=pd.read_csv(OUT/'extension.csv')
    assert len(extra)==len(json.loads((OUT/'extension_design.json').read_text())['network_ids'])*2*10
    rows=pd.concat([rows,extra],ignore_index=True)
    ranking=rows.groupby(['network','head','kind']).agg(rmse=('rmse','mean'),seed_std=('rmse','std'),
            train_rmse=('train_rmse','mean')).reset_index().sort_values(['rmse','network','head'])
    ranking.to_csv(OUT/'development_ranking.csv',index=False)
    selected=[]
    for kind in ['adam','rules','linear_rules','hinge_rules','hinge_only']:
        # Preserve two distinct network structures/configurations per family.
        for row in ranking[ranking.kind==kind].drop_duplicates('network').head(2).itertuples():
            selected.append(dict(network=int(row.network),head=int(row.head)))
    anchor=dict(network=0,head=0)
    if anchor not in selected:
        selected.append(anchor)
    frozen(OUT/'shortlist.json',selected)
    return selected


def evaluate(frame):
    selected=shortlist()
    for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=0).split(frame)):
        train,test=frame.iloc[tr],frame.iloc[te]
        path=OUT/f'inner_{fold}.csv'
        rows=pd.read_csv(path).to_dict('records') if path.exists() else []
        for k,(a,b) in enumerate(KFold(3,shuffle=True,random_state=100+fold).split(train)):
            for nid in sorted({c['network'] for c in selected}):
                ids=[i for i,c in enumerate(selected) if c['network']==nid]
                done={int(r['candidate']) for r in rows if int(r['inner_fold'])==k}
                if all(i in done for i in ids):
                    continue
                model,state,_=fit(train.iloc[a],networks()[nid],1000+fold*10+k)
                for cid in ids:
                    if cid in done:
                        continue
                    head=get_head(model,state,train.iloc[a],all_heads()[selected[cid]['head']])
                    pred=predict_refit(model,state,train.iloc[b],head)
                    rows.append(dict(candidate=cid,inner_fold=k,**scores(train.iloc[b].Rings.to_numpy(),pred)))
                pd.DataFrame(rows).to_csv(path,index=False)
                print('INNER',fold,k,nid,flush=True)
        means=pd.DataFrame(rows).groupby('candidate').rmse.mean()
        # Also report a separately selected pure-rule RRL, avoiding hybrid-only claims.
        families={'all':[i for i,c in enumerate(selected) if all_heads()[c['head']]['kind']!='hinge_only'],
                  'pure':[i for i,c in enumerate(selected) if all_heads()[c['head']]['kind'] in ['adam','rules']],
                  'hinge_only':[i for i,c in enumerate(selected) if all_heads()[c['head']]['kind']=='hinge_only']}
        for family,ids in families.items():
            prefix=OUT/f'{family}_fold_{fold}'
            if prefix.with_suffix('.csv').exists():
                continue
            cid=int(means.loc[ids].idxmin()); c=selected[cid]
            model,state,args=fit(train,networks()[c['network']],42+fold)
            head=get_head(model,state,train,all_heads()[c['head']])
            pred=predict_refit(model,state,test,head)
            graph=export(model,state,train,head,prefix)
            err=float(np.max(np.abs(pred-predict_graph(graph,test))))
            assert err<1e-4
            pd.DataFrame(dict(index=test.index,Rings=test.Rings,prediction=pred)).to_csv(prefix.with_suffix('.csv'),index=False)
            write_json(OUT/f'{family}_metrics_{fold}.json',dict(fold=fold,candidate=cid,**scores(test.Rings.to_numpy(),pred),
                export_error=err,logical_edges=0 if family=='hinge_only' else int(sum(np.sum(g['conjunction'])+np.sum(g['disjunction']) for g in graph['layers'])),
                continuous_terms=len(head['linear_weights']) if head is not None else 0,
                head_kind=all_heads()[c['head']]['kind']))
            print('OUTER',family,fold,scores(test.Rings.to_numpy(),pred)['rmse'],flush=True)


def final(frame):
    selected=shortlist()
    records=pd.concat([pd.read_csv(OUT/f'inner_{i}.csv') for i in range(5)])
    eligible=[i for i,c in enumerate(selected) if all_heads()[c['head']]['kind']!='hinge_only']
    cid=int(records.groupby('candidate').rmse.mean().loc[eligible].idxmin()); c=selected[cid]
    model,state,args=fit(frame,networks()[c['network']],2026)
    head=get_head(model,state,frame,all_heads()[c['head']])
    graph=export(model,state,frame,head,OUT/'final_rules')
    err=float(np.max(np.abs(predict_refit(model,state,frame,head)-predict_graph(graph,frame))))
    assert err<1e-4
    torch.save(dict(model_state_dict=model.net.state_dict(),state=state,rrl_args=args,head=head),OUT/'final_model.pth')
    write_json(OUT/'final_config.json',dict(candidate=cid,network=networks()[c['network']],head=all_heads()[c['head']],export_error=err))
    print('FINAL',cid,err,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['develop','extend','evaluate','final'])
    args=parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    {'develop':develop,'extend':extend,'evaluate':evaluate,'final':final}[args.stage](load_abalone())

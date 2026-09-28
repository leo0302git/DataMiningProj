"""Reproducible Abalone baselines, ablation, nested search and final fit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.eda import load_abalone
from analysis.abalone_eda import anomaly_flags
from models.manual_glm.ridge import RidgeRegression
from models.rrl.abalone import fit, predict, prepare, export_rules, predict_rules

OUT = ROOT/'results/abalone'
BASE = dict(threshold='random', structure='5@16', lr=.002, wd=.001,
            alpha=.9, beta=3, gamma=3, epochs=100, loss='mse', transform='standard', use_not=False)


def configs():
    anchors = [BASE, dict(BASE, threshold='quantile'),
               dict(BASE, threshold='supervised'), dict(BASE, threshold='quantile', loss='huber'),
               dict(BASE, threshold='quantile', transform='log'),
               dict(BASE, threshold='quantile', structure='5@32', lr=.005, wd=0., alpha=.999, beta=8, epochs=160)]
    rng = np.random.default_rng(20260928)
    while len(anchors) < 24:
        alpha, beta, gamma = [(0.9,3,3),(.999,8,1),(.999,8,3)][int(rng.integers(3))]
        config = dict(threshold=str(rng.choice(['random','quantile','supervised'])),
                      structure=str(rng.choice(['5@16','5@32','5@64','10@32','5@32@16'])),
                      lr=float(rng.choice([.001,.002,.005])), wd=float(rng.choice([0,.0001,.001])),
                      alpha=alpha, beta=beta, gamma=gamma, epochs=int(rng.choice([80,160])),
                      loss=str(rng.choice(['mse','huber'])), transform=str(rng.choice(['standard','log'])),
                      use_not=bool(rng.integers(2)))
        if config not in anchors:
            anchors.append(config)
    return anchors


def scores(y, prediction):
    assert np.isfinite(prediction).all()
    tail = y >= 20
    return dict(rmse=float(np.sqrt(mean_squared_error(y,prediction))),
                mae=float(mean_absolute_error(y,prediction)), r2=float(r2_score(y,prediction)),
                tail_n=int(tail.sum()), tail_mae=float(mean_absolute_error(y[tail],prediction[tail])) if tail.any() else None,
                bias=float(np.mean(prediction-y)), tail_bias=float(np.mean(prediction[tail]-y[tail])) if tail.any() else None)


def baseline(kind, config, train, test):
    X, state = prepare(train)
    Xt, _ = prepare(test, state)
    if kind == 'ridge':
        model = RidgeRegression(config['alpha'])
    else:
        model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=2, **config)
    model.fit(X, train.Rings.to_numpy(float))
    return model.predict(Xt)


def save_fold(name, fold, train, test, prediction, config, extra=None):
    folder = OUT/name
    folder.mkdir(parents=True, exist_ok=True)
    frame = test[['Sex','Rings']].copy()
    frame.insert(0, 'index', test.index)
    frame['prediction'] = prediction
    frame['suspect'] = anomaly_flags(test).any(axis=1)
    frame.to_csv(folder/f'fold_{fold}_predictions.csv', index=False)
    row = dict(fold=fold, train_n=len(train), **scores(test.Rings.to_numpy(), prediction), config=config, **(extra or {}))
    (folder/f'fold_{fold}_metrics.json').write_text(json.dumps(row, indent=2))
    print(name, fold, 'RMSE', round(row['rmse'],4), flush=True)


def rrl_fold(name, fold, train, test, config):
    model, state, args = fit(train, config, 42+fold)
    pred = predict(model,state,test)
    hard = predict(model,state,test,hard=True)
    folder = OUT/name
    folder.mkdir(parents=True, exist_ok=True)
    graph = export_rules(model,state,train,folder/f'fold_{fold}_rules')
    external = predict_rules(graph,test)
    edges = sum(np.sum(layer['conjunction'])+np.sum(layer['disjunction']) for layer in graph['layers'])
    extra = dict(rule_nodes=sum(len(layer['conjunction'][0])+len(layer['disjunction'][0]) for layer in graph['layers']),
                 logical_edges=int(edges), log_edges=float(np.log(max(edges,1))),
                 hard_max_error=float(np.max(np.abs(pred-hard))),
                 exported_max_error=float(np.max(np.abs(pred-external))))
    assert extra['hard_max_error'] < 1e-5 and extra['exported_max_error'] < 1e-4, extra
    save_fold(name,fold,train,test,pred,config,extra)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['initial','search','final','paired'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    frame = load_abalone()
    OUT.mkdir(parents=True, exist_ok=True)
    # Freeze search and exact data identity before any performance observations.
    manifest = dict(configs=configs(), data_sha256=hashlib.sha256((ROOT/'data/raw/abalone/abalone.data').read_bytes()).hexdigest(),
                    outer=dict(n_splits=5, shuffle=True, random_state=0), inner_folds=3,
                    inner_seed='100 + outer_fold', criterion='minimum mean inner RMSE', batch_size=128,
                    anomaly_policy='retain primary; sensitivity removes flagged training rows only')
    path = OUT/'design.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'Existing design differs; do not overwrite experiments'
    else:
        path.write_text(json.dumps(manifest, indent=2))
    splits = list(KFold(5,shuffle=True,random_state=0).split(frame))
    if args.stage == 'final':
        searches = pd.concat([pd.read_csv(OUT/f'search_fold_{i}.csv') for i in range(5)])
        ranking = searches.groupby('config_id').mean(numeric_only=True).sort_values('inner_rmse')
        best = int(ranking.index[0])
        config = configs()[best]
        model,state,rrl_args = fit(frame,config,2026)
        torch.save(dict(model_state_dict=model.net.state_dict(),rrl_args=rrl_args,state=state,config=config), OUT/'final_model.pth')
        graph = export_rules(model,state,frame,OUT/'final_rules')
        error = float(np.max(np.abs(predict(model,state,frame)-predict_rules(graph,frame))))
        assert error < 1e-4
        (OUT/'final_config.json').write_text(json.dumps(dict(config_id=best,config=config,
             mean_inner_rmse=float(ranking.iloc[0].inner_rmse), exported_max_error=error),indent=2))
        print('FINAL', config, 'export error', error, flush=True)
        return
    for fold,(tr,te) in enumerate(splits):
        train,test = frame.iloc[tr],frame.iloc[te]
        if args.stage == 'paired':
            rrl_fold('rrl_quantile_paired',fold,train,test,dict(BASE,threshold='quantile',match_random_init=True))
            continue
        inner = list(KFold(3,shuffle=True,random_state=100+fold).split(train))
        if args.stage == 'initial':
            if not (OUT/'mean'/f'fold_{fold}_metrics.json').exists():
                save_fold('mean',fold,train,test,np.full(len(test),train.Rings.mean()),{})
            for kind,pool in [('ridge',[dict(alpha=a) for a in [0,.001,.01,.1,1]]),
                              ('forest',[dict(max_depth=d,min_samples_leaf=l) for d in [8,None] for l in [1,5,10]])]:
                if (OUT/kind/f'fold_{fold}_metrics.json').exists():
                    continue
                records=[]
                for config in pool:
                    values=[scores(train.iloc[b].Rings.to_numpy(),baseline(kind,config,train.iloc[a],train.iloc[b]))['rmse'] for a,b in inner]
                    records.append(dict(config=config,inner_rmse=float(np.mean(values)),inner_scores=values))
                best=min(records,key=lambda row: row['inner_rmse'])['config']
                save_fold(kind,fold,train,test,baseline(kind,best,train,test),best)
                (OUT/kind/f'fold_{fold}_search.json').write_text(json.dumps(records,indent=2))
            for name,config,clean in [('rrl_original',BASE,False),('rrl_quantile',dict(BASE,threshold='quantile'),False),
                                       ('rrl_clean_sensitivity',BASE,True)]:
                if not (OUT/name/f'fold_{fold}_metrics.json').exists():
                    used=train.loc[~anomaly_flags(train).any(axis=1)] if clean else train
                    rrl_fold(name,fold,used,test,config)
        else:
            search_path=OUT/f'search_fold_{fold}.csv'
            records=pd.read_csv(search_path).to_dict('records') if search_path.exists() else []
            done={int(row['config_id']) for row in records}
            for cid,config in enumerate(configs()):
                if cid in done:
                    continue
                start=time.monotonic()
                values=[]
                for k,(a,b) in enumerate(inner):
                    model,state,_=fit(train.iloc[a],config,1000+fold*10+k)
                    values.append(scores(train.iloc[b].Rings.to_numpy(),predict(model,state,train.iloc[b]))['rmse'])
                records.append(dict(config_id=cid,inner_rmse=float(np.mean(values)),inner_std=float(np.std(values,ddof=1)),
                                    scores=json.dumps(values), config=json.dumps(config,sort_keys=True),seconds=time.monotonic()-start))
                pd.DataFrame(records).to_csv(search_path,index=False)
                print('search',fold,cid,'RMSE',np.mean(values),'seconds',round(time.monotonic()-start),flush=True)
            best=min(records,key=lambda row: (row['inner_rmse'],row['config_id']))
            if not (OUT/'rrl_optimized'/f'fold_{fold}_metrics.json').exists():
                rrl_fold('rrl_optimized',fold,train,test,configs()[int(best['config_id'])])


if __name__ == '__main__':
    main()

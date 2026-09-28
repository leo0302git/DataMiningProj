"""Summarize completed folds, paired ablations and out-of-fold diagnostics."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def audit():
    """Recompute metrics, exact folds, preprocessing and exported-rule predictions."""
    import sys
    import hashlib
    import torch
    from sklearn.model_selection import KFold
    sys.path.insert(0, str(ROOT))
    from analysis.eda import load_abalone
    from analysis.abalone_eda import anomaly_flags
    from experiments.run_abalone import configs, scores
    from models.rrl.abalone import prepare, predict_rules, predict, RRL
    frame=load_abalone()
    design=json.loads((OUT/'design.json').read_text())
    assert design['configs']==configs()
    assert design['data_sha256']==hashlib.sha256((ROOT/'data/raw/abalone/abalone.data').read_bytes()).hexdigest()
    folds=list(KFold(5,shuffle=True,random_state=0).split(frame))
    names=['mean','ridge','forest','rrl_original','rrl_quantile','rrl_quantile_paired','rrl_clean_sensitivity','rrl_optimized']
    max_error=0.
    threshold_rows=[]
    for name in names:
        indices=[]
        for fold,(tr,te) in enumerate(folds):
            p=pd.read_csv(OUT/name/f'fold_{fold}_predictions.csv')
            assert np.array_equal(p['index'],te)
            assert np.array_equal(p.Rings,frame.iloc[te].Rings)
            indices.extend(p['index'].tolist())
            record=json.loads((OUT/name/f'fold_{fold}_metrics.json').read_text())
            for key,value in scores(p.Rings.to_numpy(),p.prediction.to_numpy()).items():
                assert np.isclose(record[key],value), (name,fold,key)
            if name.startswith('rrl'):
                graph=json.loads((OUT/name/f'fold_{fold}_rules.json').read_text())
                train=frame.iloc[tr]
                if name=='rrl_clean_sensitivity':
                    train=train.loc[~anomaly_flags(train).any(axis=1)]
                _,state=prepare(train,transform=graph['preprocessing']['transform'])
                for key in ['mean','std','categories']:
                    assert state[key] == graph['preprocessing'][key], (name,fold,key)
                assert np.isclose(train.Rings.mean(),graph['preprocessing']['y_mean'])
                err=float(np.max(np.abs(predict_rules(graph,frame.iloc[te])-p.prediction)))
                assert err<1e-4, (name,fold,err)
                max_error=max(max_error,err)
                X,_=prepare(train,graph['preprocessing'])
                ndisc=len(state['categories'])-1
                cuts=np.asarray(graph['thresholds'],dtype='float32')
                activation=(X[:,ndisc:,None]>cuts.T).mean(axis=0)
                for j,feature in enumerate(state['features']):
                    threshold_rows.append(dict(model=name,fold=fold,feature=feature,
                         min_activation=float(activation[j].min()),max_activation=float(activation[j].max()),
                         near_constant_fraction=float(((activation[j]<.01)|(activation[j]>.99)).mean()),
                         activation_span=float(np.ptp(activation[j]))))
        assert sorted(indices)==list(range(len(frame)))
    for fold in range(5):
        search=pd.read_csv(OUT/f'search_fold_{fold}.csv')
        assert sorted(search.config_id)==list(range(24))
        for row in search.itertuples():
            assert json.loads(row.config)==configs()[row.config_id]
            assert len(json.loads(row.scores))==3
            assert np.isclose(row.inner_rmse,np.mean(json.loads(row.scores)))
        selected=json.loads((OUT/'rrl_optimized'/f'fold_{fold}_metrics.json').read_text())['config']
        best=search.sort_values(['inner_rmse','config_id']).iloc[0]
        assert selected==json.loads(best.config)
    checkpoint=torch.load(OUT/'final_model.pth',map_location='cpu',weights_only=False)
    searches=pd.concat([pd.read_csv(OUT/f'search_fold_{i}.csv') for i in range(5)])
    best_id=int(searches.groupby('config_id').inner_rmse.mean().idxmin())
    assert checkpoint['config']==configs()[best_id]
    model=RRL(**checkpoint['rrl_args'])
    model.net.load_state_dict(checkpoint['model_state_dict'])
    graph=json.loads((OUT/'final_rules.json').read_text())
    final_error=float(np.max(np.abs(predict(model,checkpoint['state'],frame)-predict_rules(graph,frame))))
    assert final_error<1e-4
    result=dict(models=len(names),folds_each=5,samples_each=len(frame),inner_fits=5*24*3,
                oof_max_rule_error=max_error, final_checkpoint_rule_error=final_error,
                verified=['data_hash','fold_indices','coverage','labels','metrics','training_preprocessing','search_configurations',
                          'inner_selection','final_selection','independent_rules','checkpoint_reload'])
    (OUT/'audit.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(threshold_rows).to_csv(OUT/'threshold_diagnostics.csv',index=False)
    print('AUDIT',json.dumps(result))

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/abalone'


def main():
    rows=[]
    predictions=[]
    for path in sorted(OUT.glob('*/fold_*_metrics.json')):
        record=json.loads(path.read_text())
        record['model']=path.parent.name
        record['config']=json.dumps(record['config'],sort_keys=True)
        rows.append(record)
    metrics=pd.DataFrame(rows)
    metrics.to_csv(OUT/'fold_metrics.csv',index=False)
    columns=['rmse','mae','r2','tail_mae','tail_bias','rule_nodes','logical_edges','log_edges','exported_max_error']
    summary=metrics.groupby('model')[columns].agg(['mean','std'])
    summary.to_csv(OUT/'summary.csv')
    print(summary[['rmse','mae','r2','tail_mae']].round(4).to_string())
    for path in sorted(OUT.glob('*/fold_*_predictions.csv')):
        frame=pd.read_csv(path)
        frame['model']=path.parent.name
        predictions.append(frame)
    prediction=pd.concat(predictions,ignore_index=True)
    prediction['residual']=prediction.prediction-prediction.Rings
    prediction['absolute_error']=prediction.residual.abs()
    prediction['ring_group']=pd.cut(prediction.Rings,[0,7,11,19,100],labels=['1-7','8-11','12-19','20+'])
    for col in ['Sex','ring_group','suspect']:
        prediction.groupby(['model',col],observed=True).agg(n=('Rings','size'),mae=('absolute_error','mean'),
            bias=('residual','mean')).to_csv(OUT/f'errors_by_{col}.csv')
    original=metrics[metrics.model=='rrl_original'].set_index('fold')
    for name in ['rrl_quantile','rrl_quantile_paired','rrl_clean_sensitivity']:
        other=metrics[metrics.model==name].set_index('fold')
        delta=other[['rmse','mae','r2','tail_mae']]-original[['rmse','mae','r2','tail_mae']]
        delta.loc['mean']=delta.mean()
        delta.to_csv(OUT/f'{name}_paired_delta.csv')
    models=[name for name in ['mean','ridge','forest','rrl_original','rrl_quantile_paired','rrl_optimized'] if name in summary.index]
    fig,axes=plt.subplots(1,3,figsize=(13,4.6))
    for ax,metric in zip(axes,['rmse','mae','r2']):
        values=summary.loc[models,metric]
        ax.bar(models,values['mean'],yerr=values['std'],capsize=3)
        ax.set_title(metric.upper()+' (fold mean +/- SD)')
        ax.tick_params(axis='x',rotation=65)
    fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=160); plt.close(fig)
    models=[m for m in models if m != 'mean']
    fig,axes=plt.subplots(len(models),3,figsize=(12,3*len(models)))
    for axes_row,name in zip(axes,models):
        p=prediction[prediction.model==name]
        axes_row[0].scatter(p.Rings,p.prediction,s=5,alpha=.2)
        axes_row[0].plot([1,29],[1,29],'k--')
        axes_row[0].set(xlabel='True Rings',ylabel='Predicted Rings',title=name)
        axes_row[1].hist(p.residual,bins=40)
        axes_row[1].axvline(0,color='black',ls='--')
        axes_row[1].set(xlabel='Prediction - Rings',title='OOF residuals')
        groups=p.groupby('Rings').residual.agg(['mean','std','size'])
        axes_row[2].errorbar(groups.index,groups['mean'],yerr=groups['std'],marker='.',capsize=2)
        axes_row[2].axhline(0,color='black',ls='--')
        axes_row[2].set(xlabel='True Rings',ylabel='Residual mean +/- SD',title='Tail groups have few samples')
    fig.tight_layout(); fig.savefig(OUT/'residuals.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for name in ['rrl_original','rrl_quantile_paired','rrl_optimized']:
        if name in summary.index:
            ax.errorbar(summary.loc[name,('log_edges','mean')],summary.loc[name,('rmse','mean')],
                 xerr=summary.loc[name,('log_edges','std')],yerr=summary.loc[name,('rmse','std')],
                 marker='o',capsize=3,label=name)
    ax.set(xlabel='log(raw logical edges)',ylabel='RMSE (lower is better)',title='Performance and rule complexity')
    ax.legend();fig.tight_layout();fig.savefig(OUT/'complexity.png',dpi=160);plt.close(fig)
    paths=sorted(OUT.glob('search_fold_*.csv'))
    if len(paths)==5:
        searches=pd.concat([pd.read_csv(p).assign(outer_fold=i) for i,p in enumerate(paths)])
        ranking=searches.groupby('config_id').agg(mean_inner_rmse=('inner_rmse','mean'),
              std_inner_rmse=('inner_rmse','std'),config=('config','first')).sort_values('mean_inner_rmse')
        ranking.to_csv(OUT/'search_ranking.csv')
        pivot=searches.pivot(index='outer_fold',columns='config_id',values='inner_rmse')
        ablations=[]
        for label,reference,changed in [('random_to_quantile_unmatched_init',0,1),('quantile_to_supervised',1,2),
                                         ('quantile_mse_to_huber',1,3),('quantile_standard_to_log',1,4)]:
            delta=pivot[changed]-pivot[reference]
            ablations.append(dict(change=label,reference_id=reference,changed_id=changed,
                    reference_rmse=pivot[reference].mean(),changed_rmse=pivot[changed].mean(),
                    delta_mean=delta.mean(),delta_std=delta.std(),improved_outer_training_splits=int((delta<0).sum())))
        pd.DataFrame(ablations).to_csv(OUT/'inner_ablation.csv',index=False)


if __name__ == '__main__':
    main()
    if (OUT/'final_model.pth').exists():
        audit()

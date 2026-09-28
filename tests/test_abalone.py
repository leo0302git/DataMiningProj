import json

import numpy as np
import torch

from analysis.eda import load_abalone
from experiments.run_abalone import BASE
from models.manual_glm.ridge import RidgeRegression
from models.rrl.abalone import fit, predict, prepare, export_rules, predict_rules
from models.rrl.abalone_refine import refit, predict_refit, graph_with_head, predict_graph, extra_features
from experiments.abalone_round3 import train_rrl, targets, networks, rf_configs


def test_hinge_basis_order_and_continuity():
    X = np.array([[0,1,-1],[1,0,.5],[0,0,2]],dtype='float32')
    basis = extra_features(X,dict(categories=['F','I','M']),[[0.],[1.]])
    assert np.array_equal(basis[:,:3],X)
    assert np.array_equal(basis[:,3:],[[0,0],[.5,0],[2,1]])


def test_ridge_and_regression_export(tmp_path):
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 2*X[:, 0] + 7
    assert np.allclose(RidgeRegression().fit(X,y).predict(X), y)
    regularized = RidgeRegression(1).fit(X,y)
    assert 0 < regularized.coef_[1] < 2

    torch.set_num_threads(1)
    frame = load_abalone()
    train, test = frame.iloc[:96], frame.iloc[100:130].copy()
    config = dict(BASE, epochs=2, structure='3@4@3', use_not=True, threshold='supervised')
    original,_,_ = fit(train,dict(BASE,epochs=0),3)
    paired,_,_ = fit(train,dict(BASE,epochs=0,threshold='quantile',match_random_init=True),3)
    for (_,a),(_,b) in zip(original.net.named_parameters(),paired.net.named_parameters()):
        assert torch.equal(a,b)
    history = []
    model,state,args = fit(train,config,3,validation=test,history=history)
    assert history[-1]['epoch'] == 2
    assert np.isfinite(history[-1]['validation_rmse'])
    assert np.isclose(state['y_mean'],train.Rings.mean())
    before = json.dumps(state,sort_keys=True)
    test['Rings'] = 1000
    test['Length'] *= 100
    prepare(test,state)
    assert json.dumps(state,sort_keys=True) == before
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.net.parameters())
    graph = export_rules(model,state,train,tmp_path/'rules')
    graph = json.loads((tmp_path/'rules.json').read_text())
    assert np.allclose(predict(model,state,test),predict_rules(graph,test),atol=1e-5)
    assert np.allclose(predict(model,state,test),predict(model,state,test,hard=True),atol=1e-5)
    restored = type(model)(**args)
    restored.net.load_state_dict(model.net.state_dict())
    assert np.array_equal(predict(model,state,test),predict(restored,state,test))
    for kind in ['rules', 'linear_rules', 'hinge_rules', 'hinge_only']:
        head = refit(model,state,train,kind,.001)
        revised = graph_with_head(graph,head)
        assert np.allclose(predict_refit(model,state,test,head),predict_graph(revised,test),atol=1e-5)
        assert np.array_equal(predict(model,state,test),predict(restored,state,test))
    unpenalized = refit(model,state,train,'rules',0.)
    assert np.mean((predict_refit(model,state,train,unpenalized)-train.Rings)**2) <= np.mean((predict(model,state,train)-train.Rings)**2)+1e-8
    ordinary=refit(model,state,train,'hinge_rules',.1)
    grouped=refit(model,state,train,'hinge_rules',.1,continuous_alpha=.1)
    assert np.allclose(predict_refit(model,state,train,ordinary),predict_refit(model,state,train,grouped),atol=1e-7)
    grouped=refit(model,state,train,'hinge_rules',1.,continuous_alpha=.01)
    assert np.allclose(predict_refit(model,state,test,grouped),predict_graph(graph_with_head(graph,grouped),test),atol=1e-5)


def test_residual_and_early_stop_isolation():
    torch.set_num_threads(1)
    frame=load_abalone();train=frame.iloc[:96];valid=frame.iloc[100:120].copy()
    config=dict(networks()[0],structure='3@4',residual=True,epochs=4,stop_patience=1,monitor_every=1)
    a,b=targets(train,valid,config)
    valid['Rings']+=1000
    changed,_=targets(train,valid,config)
    assert np.array_equal(a.Rings,changed.Rings)
    assert np.array_equal(train.Rings,frame.iloc[:96].Rings)
    model,state,_=train_rrl(train,config,5)
    assert 1 <= state['selected_epochs'] <= 4
    head=refit(model,state,train,'hinge_rules',1.,.01)
    assert np.isfinite(predict_refit(model,state,valid,head)).all()
    assert len(networks())==37 and len(rf_configs())==72

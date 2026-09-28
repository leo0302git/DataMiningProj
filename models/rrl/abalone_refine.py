"""Training-only refit of an RRL rule head, optionally with linear input skips."""
import numpy as np
import torch

from models.manual_glm.ridge import RidgeRegression
from models.rrl.abalone import prepare, predict, predict_rules


def representation(model, state, frame):
    X, _ = prepare(frame, state)
    values = torch.from_numpy(X)
    with torch.no_grad():
        for layer in model.net.layer_list[:-1]:
            values = layer.binarized_forward(values)
    return X, values.numpy().astype(float)


def refit(model, state, train, kind, alpha, continuous_alpha=None):
    X, rules = representation(model, state, train)
    knots = model.net.layer_list[0].cl.detach().numpy().tolist() if kind.startswith('hinge') else None
    continuous = extra_features(X,state,knots) if kind != 'rules' else np.empty((len(X),0))
    used_rules = np.empty((len(X),0)) if kind == 'hinge_only' else rules
    features = np.column_stack([used_rules,continuous])
    if continuous_alpha is None:
        coef = RidgeRegression(alpha).fit(features, train.Rings.to_numpy(float)).coef_
    else:
        if min(alpha,continuous_alpha) <= 0:
            raise ValueError('Grouped penalties must be positive')
        scale=np.sqrt(np.r_[np.full(used_rules.shape[1],alpha),np.full(continuous.shape[1],continuous_alpha)])
        coef=RidgeRegression(1.).fit(features/scale,train.Rings.to_numpy(float)).coef_
        coef[1:] /= scale
    return dict(kind=kind, alpha=alpha, continuous_alpha=continuous_alpha, bias=float(coef[0]),
                weights=coef[1:used_rules.shape[1]+1].tolist() if kind != 'hinge_only' else np.zeros(rules.shape[1]).tolist(),
                linear_weights=coef[used_rules.shape[1]+1:].tolist(),hinge_knots=knots)


def extra_features(X, state, knots):
    if knots is None:
        return X
    continuous=X[:,len(state['categories'])-1:]
    hinges=np.maximum(continuous[:,:,None]-np.asarray(knots,dtype='float32').T,0).reshape(len(X),-1)
    return np.column_stack([X,hinges])


def predict_refit(model, state, frame, head):
    if head is None:
        return predict(model, state, frame)
    X, rules = representation(model, state, frame)
    output = head['bias'] + rules @ np.asarray(head['weights'])
    if head['linear_weights']:
        output += extra_features(X,state,head.get('hinge_knots')) @ np.asarray(head['linear_weights'])
    return output


def graph_with_head(graph, head):
    graph = dict(graph)
    if head is not None:
        graph.update(weights=head['weights'], bias=head['bias'], linear_weights=head['linear_weights'],
                     head_kind=head['kind'], head_alpha=head['alpha'],hinge_knots=head.get('hinge_knots'))
        graph['continuous_alpha']=head.get('continuous_alpha')
    return graph


def predict_graph(graph, frame):
    if 'members' in graph:
        if graph.get('aggregation') != 'arithmetic_mean' or not graph['members']:
            raise ValueError('Expected a nonempty arithmetic-mean ensemble')
        return np.mean([predict_graph(member,frame) for member in graph['members']],axis=0)
    output = predict_rules(graph, frame)
    if graph.get('linear_weights'):
        X, _ = prepare(frame, graph['preprocessing'])
        output += extra_features(X,graph['preprocessing'],graph.get('hinge_knots')) @ np.asarray(graph['linear_weights'])
    return output

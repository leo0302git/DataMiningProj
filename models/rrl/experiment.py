import os
import json
import logging
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import defaultdict

from rrl.utils import read_csv, DBEncoder
from rrl.models import RRL

DATA_DIR = './dataset'


def get_data_loader(dataset, batch_size, k=0, pin_memory=False, save_best=True):
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=False)

    labels = y_df.iloc[:, 0].to_numpy()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    train_index, test_index = list(kf.split(X_df, labels))[k]
    X_train_df = X_df.iloc[train_index]
    y_train_df = y_df.iloc[train_index]
    X_test_df = X_df.iloc[test_index]
    y_test_df = y_df.iloc[test_index]

    X_valid_df = y_valid_df = None
    if save_best:
        X_train_df, X_valid_df, y_train_df, y_valid_df = train_test_split(
            X_train_df, y_train_df, test_size=0.05, random_state=42,
            stratify=y_train_df.iloc[:, 0])

    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_train_df, y_train_df)
    X_train, y_train = db_enc.transform(X_train_df, y_train_df, normalized=True)
    X_test, y_test = db_enc.transform(X_test_df, y_test_df, normalized=True)

    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.float32)))
    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.float32)))
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              pin_memory=pin_memory, generator=generator)
    valid_loader = None
    if save_best:
        X_valid, y_valid = db_enc.transform(X_valid_df, y_valid_df, normalized=True)
        valid_set = TensorDataset(torch.tensor(X_valid.astype(np.float32)),
                                  torch.tensor(y_valid.astype(np.float32)))
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return db_enc, train_loader, valid_loader, test_loader


def train_model(args):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(args.device)
    writer = None
    is_rank0 = True

    dataset = args.data_set
    db_enc, train_loader, valid_loader, _ = get_data_loader(
        dataset, args.batch_size, k=args.ith_kfold,
        pin_memory=device.type == 'cuda', save_best=args.save_best)

    X_fname = db_enc.X_fname
    y_fname = db_enc.y_fname
    discrete_flen = db_enc.discrete_flen
    continuous_flen = db_enc.continuous_flen
    cut_points = None
    if args.quantile_thresholds and continuous_flen:
        continuous = train_loader.dataset.tensors[0][:, discrete_flen:].numpy()
        quantiles = np.arange(1, int(args.structure.split('@')[0]) + 1)
        quantiles = quantiles / (len(quantiles) + 1)
        cut_points = torch.tensor(np.quantile(continuous, quantiles, axis=0), dtype=torch.float32)

    rrl = RRL(dim_list=[(discrete_flen, continuous_flen)] + list(map(int, args.structure.split('@'))) + [len(y_fname)],
              device_id=device,
              use_not=args.use_not,
              is_rank0=is_rank0,
              log_file=args.log,
              writer=writer,
              save_best=args.save_best,
              estimated_grad=args.estimated_grad,
              use_skip=args.skip,
              save_path=args.model,
              use_nlaf=args.nlaf,
              alpha=args.alpha,
              beta=args.beta,
              gamma=args.gamma,
              temperature=args.temp,
              cut_points=cut_points,
              distributed=False)

    rrl.train_model(
        data_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.learning_rate,
        epoch=args.epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        weight_decay=args.weight_decay,
        log_iter=args.log_iter,
        class_weights=(
            len(train_loader.dataset) /
            (2 * np.bincount(train_loader.dataset.tensors[1].numpy().argmax(axis=1)))
            if args.weighted else None))


def load_model(path, device, log_file=None, distributed=False):
    checkpoint = torch.load(path, map_location='cpu')
    saved_args = checkpoint['rrl_args']
    rrl = RRL(
        dim_list=saved_args['dim_list'],
        device_id=device,
        is_rank0=True,
        use_not=saved_args['use_not'],
        log_file=log_file,
        distributed=distributed,
        estimated_grad=saved_args['estimated_grad'],
        use_skip=saved_args['use_skip'],
        use_nlaf=saved_args['use_nlaf'],
        alpha=saved_args['alpha'],
        beta=saved_args['beta'],
        gamma=saved_args['gamma'],
        temperature=saved_args['temperature'])
    stat_dict = checkpoint['model_state_dict']
    if stat_dict and all(key.startswith('module.') for key in stat_dict):
        stat_dict = {key[7:]: value for key, value in stat_dict.items()}
    rrl.net.load_state_dict(stat_dict)
    return rrl


def test_model(args):
    rrl = load_model(args.model, torch.device(args.device), log_file=args.test_res)
    dataset = args.data_set
    db_enc, train_loader, _, test_loader = get_data_loader(
        dataset, args.batch_size, args.ith_kfold, save_best=False)
    accuracy, macro_f1 = rrl.test(test_loader=test_loader, set_name='Test')
    y_true, logits = rrl.predict(test_loader)
    _, discrete_logits = rrl.predict(test_loader, binarized=True)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    y_score = probabilities[:, 1]
    y_pred = logits.argmax(axis=1)
    if args.print_rule:
        with open(args.rrl_file, 'w') as rrl_file:
            rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, file=rrl_file, mean=db_enc.mean, std=db_enc.std)
    else:
        rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, mean=db_enc.mean, std=db_enc.std, display=False)
    
    metric = 'Log(#Edges)'
    edge_cnt = 0
    connected_rid = defaultdict(lambda: set())
    ln = len(rrl.net.layer_list) - 1
    for rid, w in rule2weights:
        connected_rid[ln - abs(rid[0])].add(rid[1])
    while ln > 1:
        ln -= 1
        layer = rrl.net.layer_list[ln]
        for r in connected_rid[ln]:
            con_len = len(layer.rule_list[0])
            if r >= con_len:
                opt_id = 1
                r -= con_len
            else:
                opt_id = 0
            rule = layer.rule_list[opt_id][r]
            edge_cnt += len(rule)
            for rid in rule:
                connected_rid[ln - abs(rid[0])].add(rid[1])
    log_edges = np.log(edge_cnt) if edge_cnt else float('-inf')
    logging.info('\n\t{} of RRL  Model: {}'.format(metric, log_edges))

    pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'positive_probability': y_score,
    }).to_csv(os.path.join(args.folder_path, 'predictions.csv'),
              index=False, float_format='%.10g')
    result = {
        'fold': args.ith_kfold,
        'accuracy': accuracy,
        'balanced_accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
        'macro_f1': macro_f1,
        'positive_precision': metrics.precision_score(y_true, y_pred, zero_division=0),
        'positive_recall': metrics.recall_score(y_true, y_pred, zero_division=0),
        'positive_f1': metrics.f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': metrics.roc_auc_score(y_true, y_score),
        'pr_auc': metrics.average_precision_score(y_true, y_score),
        'rule_count': len(rule2weights),
        'edge_count': edge_cnt,
        'log_edges': log_edges if edge_cnt else None,
        'decision_fidelity': float(np.mean(
            logits.argmax(axis=1) == discrete_logits.argmax(axis=1))),
    }
    with open(os.path.join(args.folder_path, 'metrics.json'), 'w') as output:
        json.dump(result, output, indent=2)
    return result



def train_main(args):
    train_model(args)


if __name__ == '__main__':
    from args import rrl_args
    # for arg in vars(rrl_args):
    #     print(arg, getattr(rrl_args, arg))
    train_main(rrl_args)
    test_model(rrl_args)

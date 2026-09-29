"""核查纯规则实验产物并生成中文报告；只在 final_audit 完成后运行。"""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.eda import load_abalone
from experiments.run_abalone import scores
from models.rrl.abalone import prepare, RRL
from models.rrl.abalone_refine import predict_graph, predict_refit

OUT = ROOT/'results/abalone/pure_rrl'
METRICS = ['rmse', 'mae', 'r2', 'tail_mae']
LABELS = dict(old_pure='旧纯规则流程', ridge='普通 Ridge', forest='原 RF',
              expanded_rf='扩展搜索 RF', hinge_only='无规则分段线性',
              rrl_original='原始回归 RRL', pure_314='本轮纯 RRL（314）',
              pure_2718='本轮纯 RRL（2718）')


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.5f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(cell(v) for v in row)+' |' for row in rows])


def main():
    if not (OUT/'final_audit.json').exists():
        raise RuntimeError('训练尚未完成：需要 final_audit.json，不能生成最终报告')
    design = read(OUT/'design.json')
    assert design['data_sha256'] == hashlib.sha256((ROOT/'data/raw/abalone/abalone.data').read_bytes()).hexdigest()
    for path, digest in design['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    frame = load_abalone()
    folds = list(KFold(**design['outer']).split(frame))
    seeds = design['seeds']
    candidates = design['candidates']
    records = {p.stem: read(p) for p in sorted((OUT/'trials').glob('*.json'))}
    for record in records.values():
        train, evaluation = set(record['train_indices']), set(record['evaluation_indices'])
        assert not train & evaluation
        stopping = record['stopping']
        if stopping:
            proper, stop = set(stopping['train_indices']), set(stopping['validation_indices'])
            assert not proper & stop and proper | stop == train
            assert record['selected_epoch'] in [h['epoch'] for h in stopping['history']]
    searches, fold_rows, ablations, choices, outer = [], [], [], [], []
    max_rule_error = 0.
    for fold, (tr, te) in enumerate(folds):
        inner = pd.read_csv(OUT/f'inner_{fold}.csv')
        expected = {(c, k, s) for c in range(len(candidates)) for k in range(3) for s in seeds}
        assert set(inner[['candidate', 'inner_fold', 'seed']].itertuples(index=False, name=None)) == expected
        assert len(inner) == len(expected)
        selected = read(OUT/f'selection_{fold}.json')
        assert selected['candidate'] == int(inner.groupby('candidate').rmse.mean().sort_values(kind='stable').index[0])
        assert selected['config'] == candidates[selected['candidate']]['config']
        searches.append(inner)
        choices.append([fold+1, selected['candidate'], candidates[selected['candidate']]['name'],
                        *[records[f'outer_{fold}_selected_{s}']['selected_epoch'] for s in seeds]])
        for key, record in records.items():
            if not key.startswith(f'outer_{fold}_'):
                continue
            seed = record['seed']
            variant = key[len(f'outer_{fold}_'):-len(f'_{seed}')]
            assert record['train_indices'] == frame.iloc[tr].index.tolist()
            assert record['evaluation_indices'] == frame.iloc[te].index.tolist()
            pred = pd.read_csv(OUT/f'{key}.csv')
            assert np.array_equal(pred['index'], te) and np.array_equal(pred.Rings, frame.iloc[te].Rings)
            for metric, value in scores(pred.Rings.to_numpy(), pred.prediction.to_numpy()).items():
                assert np.isclose(record[metric], value), (key, metric)
            graph = read(OUT/f'{key}.json')
            assert not graph.get('linear_weights') and not graph.get('hinge_knots') and not graph.get('gates') and 'members' not in graph
            _, state = prepare(frame.iloc[tr], transform=record['config']['transform'],
                               full_categories=record['config']['full_categories'])
            for field in state:
                assert graph['preprocessing'][field] == state[field], (key, field)
            error = float(np.max(np.abs(predict_graph(graph, frame.iloc[te])-pred.prediction)))
            assert error < 1e-6, (key, error)
            max_rule_error = max(max_rule_error, error)
            outer.append(dict(record, fold=fold, variant=variant))
            if variant == 'selected':
                fold_rows.append(dict(model=f'pure_{seed}', fold=fold, **{m:record[m] for m in METRICS}))
            else:
                base = records[f'outer_{fold}_selected_{seed}']
                ablations.append(dict(stage='outer', change=variant, seed=seed, fold=fold,
                    reference='selected', reference_rmse=base['rmse'], changed_rmse=record['rmse'],
                    rmse_delta=record['rmse']-base['rmse'],
                    mae_delta=record['mae']-base['mae'],
                    edges_delta=record['diagnostics']['logical_edges']-base['diagnostics']['logical_edges'],
                    distinct_rules_delta=record['diagnostics']['distinct_nonconstant']-base['diagnostics']['distinct_nonconstant']))
    for model, pattern in [('rrl_original', 'rrl_original/fold_{}_metrics.json'),
                           ('old_pure', 'exploration/pure_metrics_{}.json'),
                           ('ridge', 'ridge/fold_{}_metrics.json'), ('forest', 'forest/fold_{}_metrics.json'),
                           ('expanded_rf', 'round3/rf_metrics_{}.json'),
                           ('hinge_only', 'exploration/hinge_only_metrics_{}.json')]:
        for fold in range(5):
            old = read(OUT.parent/pattern.format(fold))
            fold_rows.append(dict(model=model, fold=fold, **{m:old[m] for m in METRICS}))
    metrics = pd.DataFrame(fold_rows)
    summary = metrics.groupby('model')[METRICS].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.insert(0, 'label', [LABELS[n] for n in summary.index])
    assert np.isclose(summary.loc['old_pure', 'rmse_mean'], 2.25743, atol=5e-6)

    development = pd.read_csv(OUT/'development.csv')
    assert len(development) == len(candidates)*len(seeds)
    by_name = {(c['name'], seed): records[f'dev_{cid}_{seed}']
               for cid, c in enumerate(candidates) for seed in seeds}
    pairs = [('reference', name) for name in ['full_sex', 'short_1', 'short_2', 'short_4', 'fixed_320', 'head_alpha_1']]
    pairs += [('short_2', 'full_sex_short_2'), ('short_2', 'short_2_head_alpha_1'),
              ('head_alpha_1', 'short_2_head_alpha_1')]
    for before, after in pairs:
        for seed in seeds:
            a, b = by_name[before, seed], by_name[after, seed]
            ablations.append(dict(stage='development', change=after, reference=before, seed=seed, fold=None,
                reference_rmse=a['rmse'], changed_rmse=b['rmse'], rmse_delta=b['rmse']-a['rmse'],
                mae_delta=b['mae']-a['mae'], edges_delta=b['diagnostics']['logical_edges']-a['diagnostics']['logical_edges'],
                distinct_rules_delta=b['diagnostics']['distinct_nonconstant']-a['diagnostics']['distinct_nonconstant']))
    paired = pd.DataFrame(ablations)
    audit = read(OUT/'final_audit.json')
    selection = read(OUT/'final_selection.json')
    expected_final = int(pd.concat(searches).groupby('candidate').rmse.mean().sort_values(kind='stable').index[0])
    assert selection['candidate'] == expected_final and selection['seed'] == design['delivery_seed'] == 314
    checkpoint = torch.load(OUT/'final_model.pth', map_location='cpu', weights_only=False)
    assert checkpoint['config'] == selection['config'] == candidates[expected_final]['config']
    assert checkpoint['head']['kind'] == 'rules'
    model = RRL(**checkpoint['rrl_args'])
    model.net.load_state_dict(checkpoint['model_state_dict'])
    graph = read(OUT/'final_rules.json')
    assert not graph.get('linear_weights') and not graph.get('hinge_knots') and not graph.get('gates') and 'members' not in graph
    reload_error = float(np.max(np.abs(predict_refit(model, checkpoint['state'], frame, checkpoint['head'])-predict_graph(graph, frame))))
    assert reload_error < 1e-6 and audit['pure_rule_model'] and audit['independent_test_performance'] is None
    reproduction = [read(OUT/f'reproduction_{f}.json') for f in range(5)]
    reproduction_error = max(r['prediction_error'] for r in reproduction)
    assert reproduction_error < 1e-5
    independent_audit = read(OUT/'audit.json') if (OUT/'audit.json').exists() else None
    if independent_audit is not None:
        assert independent_audit['passed']
    audit_note = ('独立审计 [audit.json](../results/abalone/pure_rrl/audit.json) 已通过。'
                  if independent_audit else '独立审计 audit.json 尚未产出。')

    summary.to_csv(OUT/'comparison.csv')
    paired.to_csv(OUT/'paired_ablation.csv', index=False)
    order = ['rrl_original', 'old_pure', 'pure_314', 'pure_2718', 'ridge', 'forest', 'expanded_rf', 'hinge_only']
    english = ['Original RRL', 'Previous pure RRL', 'Pure RRL: 314', 'Pure RRL: 2718', 'Ridge', 'Original RF', 'Expanded RF', 'No-rule hinges']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    colors = ['#aab2bd', '#7f8c8d', '#2364aa', '#59a5d8', '#ab87ff', '#d17a22', '#edb458', '#66a182']
    for ax, metric, title in zip(axes, ['rmse', 'tail_mae'], ['RMSE', 'MAE on Rings >= 20']):
        part = summary.loc[order]
        ax.bar(np.arange(len(order)), part[metric+'_mean'], yerr=part[metric+'_std'], color=colors, capsize=3)
        ax.set_xticks(np.arange(len(order)), english, rotation=55, ha='right')
        ax.set(ylabel='Rings (lower is better)', title=title+'; fold mean +/- SD')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Adaptive internal evaluation — seeds reported separately, no prediction averaging')
    fig.savefig(OUT/'comparison.png', dpi=160)
    plt.close(fig)

    summary_rows = [[LABELS[name], *[f'{summary.loc[name,m+"_mean"]:.5f} ± {summary.loc[name,m+"_std"]:.5f}' for m in METRICS]] for name in order]
    fold_table = [[LABELS[r.model], r.fold+1, r.rmse, r.mae, r.r2, r.tail_mae]
                  for r in metrics[metrics.model.str.startswith('pure_')].itertuples()]
    contrasts = []
    conclusions = []
    for seed in seeds:
        current = metrics[metrics.model == f'pure_{seed}'].set_index('fold')
        for reference in ['old_pure', 'forest', 'expanded_rf', 'hinge_only']:
            baseline = metrics[metrics.model == reference].set_index('fold')
            delta = current.rmse-baseline.rmse
            contrasts.append([seed, LABELS[reference], float(delta.mean()), int((delta < 0).sum())])
        old_delta = summary.loc[f'pure_{seed}', 'rmse_mean']-summary.loc['old_pure', 'rmse_mean']
        rf_delta = summary.loc[f'pure_{seed}', 'rmse_mean']-summary.loc['expanded_rf', 'rmse_mean']
        conclusions.append(f'种子 {seed} 相对旧纯规则流程的平均 RMSE 差为 {old_delta:+.5f}，相对扩展 RF 为 {rf_delta:+.5f}（负值更好）。')
    better_old = sum(summary.loc[f'pure_{s}', 'rmse_mean'] < summary.loc['old_pure', 'rmse_mean'] for s in seeds)
    better_rf = sum(summary.loc[f'pure_{s}', 'rmse_mean'] < summary.loc['expanded_rf', 'rmse_mean'] for s in seeds)
    conclusions.append(f'两个种子中有 {better_old} 个改善旧纯规则流程，有 {better_rf} 个低于扩展 RF；其余结果同样保留。')
    dev_rows = []
    for cid, candidate in enumerate(candidates):
        rows = [by_name[candidate['name'], s] for s in seeds]
        dev_rows.append([cid, candidate['name'], *[r['rmse'] for r in rows],
                        np.mean([r['train_rmse'] for r in rows]),
                        '/'.join(str(r['selected_epoch']) for r in rows),
                        np.mean([r['diagnostics']['distinct_nonconstant'] for r in rows]),
                        np.mean([r['diagnostics']['logical_edges'] for r in rows])])
    dev_pairs = []
    for (reference, changed), part in paired[paired.stage == 'development'].groupby(['reference', 'change'], sort=False):
        part = part.set_index('seed')
        dev_pairs.append([reference+' → '+changed, *[part.loc[s, 'rmse_delta'] for s in seeds],
                          float(part.edges_delta.mean()), float(part.distinct_rules_delta.mean())])
    lengths = [np.mean([by_name[name, s]['diagnostics']['mean_literals'] for s in seeds])
               for name in ['reference', 'short_1', 'short_2']]
    nonconstant = [np.mean([by_name[name, s]['diagnostics']['nonconstant'] for s in seeds])
                   for name in ['reference', 'short_1', 'short_2']]
    short4_initial = '/'.join(str(by_name['short_4', s]['diagnostics']['initial_nonconstant']) for s in seeds)
    outer_pairs = []
    for (variant, seed), part in paired[paired.stage == 'outer'].groupby(['change', 'seed'], sort=False):
        outer_pairs.append([variant, seed, len(part), float(part.rmse_delta.mean()),
                            int((part.rmse_delta > 0).sum()), float(part.mae_delta.mean()),
                            float(part.edges_delta.mean())])
    earlystop = paired[(paired.stage == 'outer') & (paired.change == 'no_earlystop')]
    earlystop_facts = '；'.join(f'种子 {s} 平均下降 {earlystop[earlystop.seed == s].rmse_delta.mean():.5f}，'
                               f'{int((earlystop[earlystop.seed == s].rmse_delta > 0).sum())}/5 折改善' for s in seeds)
    mechanism = []
    for seed in seeds:
        selected = [r for r in outer if r['seed'] == seed and r['variant'] == 'selected']
        for r in selected:
            d = r['diagnostics']
            mechanism.append([seed, r['fold']+1, d['initial_nonconstant'], d['nonconstant'],
                              d['distinct_nonconstant'], d['constant_false']+d['constant_true'],
                              d['logical_edges'], d['mean_literals'], d['net_gate_change_fraction'],
                              r['train_rmse'], r['rmse']])
    cost_rows = []
    for stage in ['dev', 'inner', 'outer']:
        subset = [r for key, r in records.items() if key.startswith(stage+'_')]
        cost_rows.append([stage, len(subset), sum(1+bool(r['stopping']) for r in subset),
                          sum(r['selected_epoch']+(r['stopping']['epochs_run'] if r['stopping'] else 0) for r in subset),
                          sum(r['seconds'] for r in subset)/3600])
    d = audit['diagnostics']
    text = f'''# Abalone：纯规则 RRL 新一轮优化

本轮模型的全部预测信息经过二值条件和 AND/OR 规则，输出为 $\\hat y=b+\\sum_j a_jR_j(x)$。训练保留 NLAF 与 Gradient Grafting，规则固定后只对规则激活拟合 Ridge 输出系数。两个训练种子分别报告，不平均两个模型的预测；交付种子事先固定为 314。第二至第四轮“规则＋连续/hinge”的混合模型另作历史旁支，本轮成绩不能与其混称。

## 1. 结果事实与适用边界

{table(['方法', 'RMSE', 'MAE', 'R²', 'Rings≥20 MAE'], summary_rows)}

表中为相同外层五折的指标算术均值 ± 折间样本标准差，不是置信区间；尾部每折样本少，不能只用全局 RMSE 评价。旧纯规则流程的 2.25743 已按原逐折配置重现。原 RF、扩展 RF、普通 Ridge 和无规则分段线性均读取原始结果，不因本轮表现调整其配置或分数。

![纯规则结果及强基线](../results/abalone/pure_rrl/comparison.png)

{table(['当前训练种子', '参照', '平均 RMSE 差：本轮−参照', '本轮更好折数/5'], contrasts)}

{''.join(conclusions)}这些是预测结果的比较，不能由差值断言某一模块贡献了多少。两种子共享数据、划分与选参流程，不是十个独立样本；仅一个种子改善时应报告初始化敏感性，不能只展示较好的种子。

本次实际结果中，两个种子均在五折全部优于旧纯规则流程，但也均在五折全部弱于原 RF 和扩展 RF。因此本轮确实缩小了纯 RRL 的差距，尚未超过随机森林。

{table(['方法', '外层折', 'RMSE', 'MAE', 'R²', '尾部 MAE'], fold_table)}

## 2. 从观察、假设到实现

此前扩大纯 RRL 容量仍落后强基线；混合路线又显示连续分段项提供了很强的预测力。因此本轮回到规则内部：一是原 Sex 删除 F，而首逻辑层没有显式 NOT，完整 F/I/M 词表可能让组合更直接；二是原逻辑连接都低于 0.5，初始 AND 恒真、OR 恒假，短规则初始化可能更快提供有区分力的表示。这些是待检验假设，不能把原初始化说成断梯度错误。阈值是训练前生成的 buffer，不由 Adam 学习；本轮实际冻结的候选沿用分位数阈值，没有把局部监督切点接口的存在写成已验证收益。沿用旧锚点的 log 预处理仅为控制变量，不再写成由 EDA 推导并验证有效的优化。

配对初始化按公共输入条件名称分别使用稳定随机流，输出层与批次顺序独立；添加 F 不移动已有 I/M、连续条件或输出初值。短规则从共享 drop-first 词表选择不同原始特征的 1/2/4 个条件，避免同一特征的矛盾字面量；完整词表新增 F 初始保留小正值、之后可学习。未选连接不是永久屏蔽。不同测量高度相关，所以逻辑上不矛盾的组合仍可能在训练样本中恒真或恒假，须由支持度诊断检验。

输出始终只有规则项。早停每次在训练子集拟合纯规则 Ridge 头，再用停止验证子集评价；确定轮数后，用完整当前训练集合重训规则并重拟合头。这样停止准则与交付预测流程一致。候选同时保留固定 320 轮及输出头正则强度对照，但不能只因为最终配置含某项就宣称该项起主要作用。

冻结设计见 [design.json](../results/abalone/pure_rrl/design.json)。共 {len(candidates)} 组候选全部进入每个外层训练集合的三折内部验证，两种子共 {sum(len(s) for s in searches)} 条内部评分；按六次内部 RMSE 均值选配置，外层标签只评价。开发划分只作诊断，不用于过滤候选。初始方案曾计划开发筛选，在任何外层训练/得分前修正为全部候选进入内层，初稿另存 design_initial.json。早停的 20% 子集还在每次训练集合内部，不能把它与外层测试折混同。

## 3. 开发阶段的独立对照（包括失败尝试）

{table(['ID', '候选', 'RMSE 314', 'RMSE 2718', '训练 RMSE 均值', '停止轮数314/2718', '不同非常量规则均值', '逻辑边均值'], dev_rows)}

{table(['单项对照', 'ΔRMSE 314', 'ΔRMSE 2718', 'Δ逻辑边均值', 'Δ不同非常量规则均值'], dev_pairs)}

此表的差值为“箭头右侧减左侧”，负的 RMSE 差表示改动有益。两个种子同时改善才比单个最优分数更支持方向；规则数量/长度变化提供机制线索，但不能代替验证误差。开发集已在历史探索中使用，因此这些对照是诊断证据，不能包装成独立确认；所有候选均保留，未按外层结果补充方案。

独立诊断中，short_1/short_2 在两个种子都小幅改善开发 RMSE，但最终平均条件数从参照的 {lengths[0]:.2f} 增至 {lengths[1]:.2f}/{lengths[2]:.2f}，非常量规则数均值从 {nonconstant[0]:.1f} 降至 {nonconstant[1]:.1f}/{nonconstant[2]:.1f}。因此“短初始化改善预测”得到局部支持，“最终更简洁、有效规则更多”的解释没有得到支持。完整 Sex 词表在两个种子都退步，与 short_2 组合也没有稳定增量；不能因为表达形式更直接就认定泛化更好。short_4 初始非常量规则只有 {short4_initial} 个，开发两个种子均退步，提示相关测量使结构上一致的多条件组合仍可能缺少样本支持。提高规则头 alpha 到 1 后训练和验证误差都更差，与更强收缩造成欠拟合的解释一致；但 alpha 同时改变早停轮数和最终规则，不能称为同一固定规则上的纯正则化效应。

## 4. 外层选参、去项消融与规则机制

{table(['外层折', '候选ID', '名称', '轮数314', '轮数2718'], choices)}

{table(['对照版本', '种子', '适用折数', 'ΔRMSE 对照−选中方案', '选中方案更好折数', 'ΔMAE', 'Δ逻辑边'], outer_pairs)}

reference 是整套配对参照，不是单项消融；no_earlystop 固定训练 320 轮；drop_first、empty_init、head_alpha_01 分别撤去完整词表、短规则初始化或更强规则头正则化。后几项仅在被选配置包含相应改变时生成，适用折数不足五折不能隐藏。消融是固定所选配置后去一项、按同一规则重新选停止轮数，不是为每个去项方案另做完整超参搜索。差值为“对照减选中方案”，正值才支持保留该项；方向和开发表的正负定义相反，CSV保留完整配对记录便于复核。

本轮最一致的正向消融证据来自早停：相对同配置固定训练 320 轮，{earlystop_facts}，合计 {int((earlystop.rmse_delta > 0).sum())}/{len(earlystop)} 个配对更好。这支持控制训练时长，而不能把相对历史流程的全部改善都归给早停。完整词表只在一折被选中，去掉它在两个种子都更好；短规则也只在一折被选中，该折去掉它会退步，支持范围有限。全数据最终选中的仍是 reference，未采用完整词表或短规则初始化，因此不能将最终交付描述为已经验证并采纳这些核心改进。

{table(['种子', '折', '初始非常量', '最终非常量', '不同非常量', '常量数', '边数', '平均条件数', '硬边净变化率', '训练 RMSE', '外层 RMSE'], mechanism)}

Support 是规则在当前训练集合中的激活比例；0/1 支持度为训练样本上的常量规则。“不同非常量”按训练激活向量计数，不证明逻辑全局等价；硬边净变化率比较初始和最终选边状态，不等于训练全过程跨门槛次数。短规则只约束初始化，训练仍可开启新边，不能因此声称最终规则更短；最终长度与复杂度由表中实际结果判断。只有误差与机制同时符合预期，才支持具体解释，不能把规则越多直接当成优化成功。

## 5. 最终单模型、核查与成本

最终候选为 {selection['candidate']}（{candidates[selection['candidate']]['name']}），按全部外层训练集合的内部评分均值选择，固定种子 {selection['seed']}，全数据重训轮数 {audit['selected_epoch']}。配置为 `{json.dumps(selection['config'], ensure_ascii=False, sort_keys=True)}`。最终含 {d['rule_nodes']} 个逻辑节点、{d['nonconstant']} 个训练非常量规则、{d['distinct_nonconstant']} 个不同非常量激活模式、{d['logical_edges']} 条逻辑边，平均每节点 {d['mean_literals']:.3f} 个条件。全数据模型没有独立测试成绩；上面的五折表评价的是内层选参流程，不能冒充最终 checkpoint 的外部验证。

保存于 [final_model.pth](../results/abalone/pure_rrl/final_model.pth)、[全精度规则 JSON](../results/abalone/pure_rrl/final_rules.json) 和 [可读规则](../results/abalone/pure_rrl/final_rules.md)。独立 NumPy 规则执行与全部新外层预测的最大绝对差为 {max_rule_error:.3g}，最终 checkpoint/JSON 的最大差为 {reload_error:.3g}；旧纯规则逐折复现最大预测差为 {reproduction_error:.3g}。本报告生成时还核查数据/源文件哈希、每个内部候选×折×种子的完整性、内层选优、训练与评价索引隔离、停止子集归属、外层指标重算、训练预处理和纯规则图结构。

本轮 14 项 pytest 检查已全部通过，覆盖旧路径兼容、新类别词表、初始化配对、短规则一致性、阈值边界、停止流程及导出/重载。{audit_note}

{table(['阶段', '评分记录数', '规则网络训练次数', '累计训练轮数', '记录用时/小时'], cost_rows)}

训练次数包含早停试训后的完整集合重训；累计轮数同样计入两段。记录用时为各 trial 的实际耗时之和，包含该 trial 的输出头拟合与导出，未包括历史复现、全数据最终训练、报告生成及中断等待；没有运行时价格数据，不把机器时间换算成货币费用。逐次配置、训练索引、停止曲线和耗时保留在 trials/。运行入口为 `python experiments/abalone_pure_rrl.py all`；训练结束后执行 `python analysis/abalone_pure_report.py` 重新核查并生成本报告。

本轮重新聚焦纯规则表示，但仍反复研究同一份 Abalone 数据。当前成绩属于自适应内部评价，换训练种子或重做同一五折不能构成独立确认；不能承诺纯 RRL 必须超过 RF，也不能只保留有利尝试。第二至第四轮混合模型的结果和失败对照完整保留，参见 [第二轮](ABALONE-EXPLORATION.md)、[第三轮](ABALONE-ROUND3.md)；本轮相应方案见 [纯 RRL 计划](ABALONE-RRL-PLAN.md)。
'''
    (ROOT/'docs/ABALONE-PURE-RRL.md').write_text(text)
    print(summary.round(5).to_string())
    print(f'AUDIT: rule error={max_rule_error:.3g}; final error={reload_error:.3g}; reproduction={reproduction_error:.3g}')


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()

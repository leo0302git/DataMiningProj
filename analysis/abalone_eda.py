"""Abalone evidence tables supplementing the six existing EDA figures."""
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.eda import load_abalone

OUT = Path(__file__).resolve().parents[1] / 'results/eda/abalone'


def anomaly_flags(frame):
    return pd.DataFrame({
        'nonpositive_height': frame.Height <= 0,
        'height_above_length': frame.Height > frame.Length,
        'meat_above_whole': frame['Shucked weight'] > frame['Whole weight'],
        'shell_above_whole': frame['Shell weight'] > frame['Whole weight'],
        'viscera_above_whole': frame['Viscera weight'] > frame['Whole weight'],
    }, index=frame.index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_abalone()
    features = list(frame.columns[1:-1])
    numeric = frame[features + ['Rings']]
    flags = anomaly_flags(frame)
    pd.concat([frame, flags], axis=1).loc[flags.any(axis=1)].to_csv(OUT/'suspect_records.csv', index_label='row_index')
    quality = dict(samples=len(frame), missing=int(frame.isna().sum().sum()),
                   nonfinite=int((~np.isfinite(numeric)).sum().sum()),
                   duplicate_rows=int(frame.duplicated().sum()),
                   duplicate_inputs=int(frame.drop(columns='Rings').duplicated().sum()),
                   suspect_union=int(flags.any(axis=1).sum()),
                   flags={c: int(flags[c].sum()) for c in flags},
                   tail_count=int((frame.Rings >= 20).sum()))
    (OUT/'quality.json').write_text(json.dumps(quality, indent=2))
    rows = []
    for group, subset in [('all', frame), *list(frame.groupby('Sex'))]:
        for name in numeric.columns:
            s = subset[name]
            q1, q3 = s.quantile([.25, .75])
            rows.append(dict(group=group, feature=name, n=len(s), mean=s.mean(),
                median=s.median(), mode=s.mode().iloc[0], std=s.std(), skew=s.skew(),
                min=s.min(), q01=s.quantile(.01), q25=q1, q75=q3, q99=s.quantile(.99),
                max=s.max(), iqr=q3-q1, zero_count=(s == 0).sum(),
                iqr_outliers=((s < q1-1.5*(q3-q1)) | (s > q3+1.5*(q3-q1))).sum()))
    pd.DataFrame(rows).to_csv(OUT/'group_statistics.csv', index=False)
    numeric.corr().to_csv(OUT/'pearson.csv')
    numeric.corr(method='spearman').to_csv(OUT/'spearman.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    for ax, method in zip(axes, ['pearson', 'spearman']):
        corr = numeric.corr(method=method)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
        ax.set_yticks(range(len(corr)), corr.columns)
        ax.set_title(method.title())
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.7)
    fig.savefig(OUT/'correlation_comparison.png', dpi=160)
    plt.close(fig)
    sensitivity = pd.DataFrame({
        'mean_all': frame[features].mean(), 'std_all': frame[features].std(),
        'mean_unflagged': frame.loc[~flags.any(axis=1), features].mean(),
        'std_unflagged': frame.loc[~flags.any(axis=1), features].std()})
    sensitivity['std_ratio'] = sensitivity.std_all/sensitivity.std_unflagged
    sensitivity.to_csv(OUT/'scale_sensitivity.csv')
    trend_rows = []
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, name in zip(axes.flat, features):
        for sex, subset in frame.groupby('Sex'):
            bins = pd.qcut(subset[name], 8, duplicates='drop')
            agg = subset.groupby(bins, observed=True).agg(x=(name, 'mean'), mean=('Rings','mean'),
                         std=('Rings','std'), n=('Rings','size'))
            ax.errorbar(agg.x, agg['mean'], yerr=agg['std'], marker='.', alpha=.7, label=sex)
            for row in agg.itertuples():
                trend_rows.append(dict(feature=name, sex=sex, x=row.x, rings_mean=row.mean,
                                       rings_std=row.std, n=row.n))
        ax.set(xlabel=name, ylabel='Rings: mean +/- SD')
        ax.legend()
    axes.flat[-1].axis('off')
    fig.suptitle('Within-Sex quantile-bin trends (SD is spread, not uncertainty)')
    fig.tight_layout()
    fig.savefig(OUT/'conditional_trends.png', dpi=160)
    plt.close(fig)
    pd.DataFrame(trend_rows).to_csv(OUT/'conditional_trends.csv', index=False)
    blocks = frame.assign(block=np.arange(len(frame))//500).groupby('block').agg(
        n=('Rings','size'), rings_mean=('Rings','mean'), rings_std=('Rings','std'),
        infant_fraction=('Sex',lambda x: (x == 'I').mean()))
    blocks.to_csv(OUT/'row_order_blocks.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(frame.Length, frame.Height, s=6, alpha=.3)
    axes[0].scatter(frame.loc[flags.any(axis=1),'Length'], frame.loc[flags.any(axis=1),'Height'], color='red', s=25)
    axes[0].set(xlabel='Length', ylabel='Height', title='Flagged observations (red)')
    axes[1].plot(blocks.index, blocks.rings_mean, marker='o')
    axes[1].set(xlabel='Original row block (500 rows)', ylabel='Mean Rings', title='Row-order heterogeneity')
    axes[2].bar(['<8','8-11','12-19','>=20'], [int((frame.Rings<8).sum()), int(frame.Rings.between(8,11).sum()), int(frame.Rings.between(12,19).sum()), quality['tail_count']])
    axes[2].set(ylabel='Samples', title='Target support')
    fig.tight_layout()
    fig.savefig(OUT/'quality_and_support.png', dpi=160)
    plt.close(fig)
    print(json.dumps(quality, indent=2))
    print(sensitivity.round(4).to_string())
    print(blocks.round(3).to_string())


if __name__ == '__main__':
    main()

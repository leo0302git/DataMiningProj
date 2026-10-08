"""Small, consistently styled figures used by the final course report."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.eda import load_abalone, load_dia
from analysis.abalone_eda import anomaly_flags


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
NAVY, TEAL, CORAL, GOLD = "#263F55", "#287F86", "#C65D50", "#C28B37"
GRAY, GRID = "#687681", "#E4E9EC"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": NAVY,
    "text.color": NAVY,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def grouped_box(ax, values, labels, colors):
    artists = ax.boxplot(values, tick_labels=labels, patch_artist=True, widths=0.48,
                         showfliers=False, medianprops={"color": NAVY, "linewidth": 1.6})
    for patch, color in zip(artists["boxes"], colors):
        patch.set(facecolor=color, alpha=0.55, edgecolor=color)
    for part in ("whiskers", "caps"):
        for artist in artists[part]:
            artist.set(color=GRAY, linewidth=0.9)
    ax.grid(axis="y")


def dia_figures():
    frame = load_dia()
    assert len(frame) == 597 and frame.shape[1] == 198
    negative, positive = (frame.loc[frame.Label == c] for c in (0, 1))

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.1, 2.65), layout="constrained")
    counts = [len(negative), len(positive)]
    left.bar(["Negative", "Positive"], counts, color=[NAVY, CORAL], width=0.5)
    for x, count in enumerate(counts):
        left.text(x, count + 8, f"{count} ({count/len(frame):.1%})", ha="center", fontsize=9)
    left.set_ylim(0, 520)
    left.set(ylabel="Samples", title="Class balance")
    left.grid(axis="y")
    grouped_box(right, [negative.MolMR, positive.MolMR], ["Negative", "Positive"], [NAVY, CORAL])
    right.set(ylabel="MolMR", title="A typical descriptor: MolMR")
    save(fig, "dia_profile.pdf")

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.1, 2.65), layout="constrained")
    for ax, name, title in [(left, "FractionCSP3", "FractionCSP3: broad values"),
                             (right, "PEOE_VSA2", "PEOE_VSA2: class shift")]:
        grouped_box(ax, [negative[name], positive[name]], ["Negative", "Positive"],
                    [NAVY, CORAL])
        ax.set(ylabel=name, title=title)
    assert negative.PEOE_VSA2.median() < positive.PEOE_VSA2.median()
    save(fig, "dia_box_comparison.pdf")

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.1, 2.65), layout="constrained")
    bins = np.linspace(np.log10(frame.Ipc.min()), np.log10(frame.Ipc.max()), 32)
    for subset, label, color in [(negative, "Negative", NAVY), (positive, "Positive", CORAL)]:
        left.hist(np.log10(subset.Ipc), bins=bins, density=True, histtype="step", linewidth=1.65,
                  label=label, color=color)
        q = np.linspace(0, 1, 101)
        right.plot(q, np.quantile(subset.Ipc, q), label=label, color=color, linewidth=1.7)
    left.set(xlabel=r"$\log_{10}(\mathrm{Ipc})$", ylabel="Density", title="Long-tailed descriptor")
    right.set(xlabel="Quantile", ylabel="Ipc", yscale="log", title="Ipc empirical quantiles")
    for ax in (left, right):
        ax.grid(axis="y")
    left.legend(frameon=False, fontsize=8)
    save(fig, "dia_tail.pdf")

    features = frame.drop(columns=["Label", "SMILES"])
    features = features.loc[:, features.nunique() > 1]
    corr = features.corr().to_numpy()
    pairs = np.abs(corr[np.triu_indices(len(features.columns), k=1)])
    label_corr = features.corrwith(frame.Label).sort_values()
    top = label_corr.reindex(label_corr.abs().sort_values(ascending=False).head(8).index).sort_values()
    assert len(pairs) == 15931 and int((pairs >= 0.95).sum()) == 101
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.1, 3.1), layout="constrained")
    left.hist(pairs, bins=np.linspace(0, 1, 31), color=TEAL, edgecolor="white", linewidth=0.3)
    left.axvline(0.95, color=CORAL, linestyle="--", linewidth=1.2)
    left.set(xlabel=r"Absolute descriptor correlation $|r|$", ylabel="Feature pairs",
             title="All 15,931 descriptor pairs")
    right.barh(top.index, top.values, color=[CORAL if v > 0 else NAVY for v in top.values], height=0.62)
    right.axvline(0, color=GRAY, linewidth=0.7)
    right.set(xlabel="Pearson r with label", title="Eight largest label correlations")
    for ax in (left, right):
        ax.grid(axis="x", alpha=0.65)
    save(fig, "dia_correlations.pdf")

    selected = ["Label", "fr_aniline", "SlogP_VSA10", "PEOE_VSA2", "fr_C_O",
                "NHOHCount", "fr_NH2", "FractionCSP3"]
    matrix = frame[selected].corr().to_numpy()
    assert matrix.shape == (8, 8) and np.isclose(matrix[1, 2], 0.724, atol=0.01)
    assert corr.shape == (179, 179) and np.isfinite(corr).all()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("report-dia", [CORAL, "#FFFFFF", TEAL])
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.1, 4.1), layout="constrained",
                                     gridspec_kw={"width_ratios": [1.5, 1]})
    image = left.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
    left.set_xticks(range(len(selected)), selected, rotation=45, ha="right", fontsize=9)
    left.set_yticks(range(len(selected)), selected, fontsize=9)
    left.set_title("Selected descriptors and label")
    for i in range(len(selected)):
        for j in range(len(selected)):
            value = matrix[i, j]
            left.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) >= 0.6 else NAVY)
    right.imshow(corr, cmap=cmap, vmin=-1, vmax=1, interpolation="none")
    right.set_xticks([])
    right.set_yticks([])
    right.set_title("All 179 descriptors")
    fig.colorbar(image, ax=[left, right], shrink=0.68, label="Pearson r")
    save(fig, "dia_heatmap.pdf")


def abalone_figures():
    frame = load_abalone()
    assert len(frame) == 4177 and int((frame.Rings >= 20).sum()) == 62
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.1, 2.65), layout="constrained")
    left.hist(frame.Rings, bins=np.arange(0.5, 30.5, 1), color=TEAL, edgecolor="white", linewidth=0.3)
    left.axvspan(19.5, 29.5, color=CORAL, alpha=0.18)
    left.set(xlabel="Rings", ylabel="Samples", title="Target distribution; 62 rings >= 20")
    grouped_box(right, [frame.loc[frame.Sex == sex, "Rings"] for sex in "FIM"],
                ["Female", "Infant", "Male"], [CORAL, TEAL, NAVY])
    right.set(ylabel="Rings", title="Rings by sex")
    save(fig, "abalone_profile.pdf")

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.1, 2.65), layout="constrained")
    for ax, name in [(left, "Length"), (right, "Shell weight")]:
        grouped_box(ax, [frame.loc[frame.Sex == sex, name] for sex in "FIM"],
                    ["Female", "Infant", "Male"], [CORAL, TEAL, NAVY])
        ax.set(ylabel=name, title=f"{name} by sex")
    assert frame.loc[frame.Sex == "I", "Length"].median() < frame.loc[frame.Sex == "F", "Length"].median()
    save(fig, "abalone_measurements.pdf")

    q = np.linspace(0, 1, 201)
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.1, 2.7), layout="constrained")
    for sex, color in [("F", CORAL), ("I", TEAL), ("M", NAVY)]:
        left.plot(q, np.quantile(frame.loc[frame.Sex == sex, "Rings"], q),
                  color=color, linewidth=1.55, label=sex)
    left.set(xlabel="Quantile", ylabel="Rings", title="Target quantiles by sex")
    left.legend(title="Sex", frameon=False, ncol=3, fontsize=8, title_fontsize=8)
    right.plot(q, np.quantile(frame.Height, q), color=TEAL, linewidth=1.7)
    right.scatter([0.99, 1], [frame.Height.quantile(0.99), frame.Height.max()],
                  color=CORAL, s=20, zorder=3)
    right.set(xlabel="Quantile", ylabel="Height", title="Height: extreme upper tail")
    for ax in (left, right):
        ax.grid(axis="y")
    assert frame.Height.quantile(0.99) < 0.23 and frame.Height.max() == 1.13
    save(fig, "abalone_quantiles.pdf")

    flags = anomaly_flags(frame)
    height_bad = flags.nonpositive_height | flags.height_above_length
    meat_bad = flags.meat_above_whole
    assert int(flags.any(axis=1).sum()) == 7 and int(height_bad.sum()) == 3
    assert int(meat_bad.sum()) == 4 and (height_bad | meat_bad).equals(flags.any(axis=1))
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.1, 2.85), layout="constrained")
    left.scatter(frame.Length, frame.Height, s=7, color=TEAL, alpha=0.28, linewidths=0)
    left.scatter(frame.loc[height_bad, "Length"], frame.loc[height_bad, "Height"],
                 s=38, facecolor=CORAL, edgecolor="white", linewidth=0.55, zorder=3)
    left.plot([0, 0.8], [0, 0.8], "--", color=GRAY, linewidth=0.9)
    left.set(xlabel="Length", ylabel="Height", title="Height checks")
    right.scatter(frame["Whole weight"], frame["Shucked weight"],
                  s=7, color=TEAL, alpha=0.28, linewidths=0)
    right.scatter(frame.loc[meat_bad, "Whole weight"],
                  frame.loc[meat_bad, "Shucked weight"], s=38, facecolor=CORAL,
                  edgecolor="white", linewidth=0.55, zorder=3)
    right.plot([0, 0.8], [0, 0.8], "--", color=GRAY, linewidth=0.9)
    right.set(xlim=(0, 0.8), ylim=(0, 0.8), xlabel="Whole weight", ylabel="Shucked weight",
              title="Meat-weight checks (low-weight detail)")
    for ax in (left, right):
        ax.grid(alpha=0.6)
    save(fig, "abalone_checks.pdf")

    names = ["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
             "Viscera weight", "Shell weight", "Rings"]
    matrix = frame[names].corr().to_numpy()
    short = ["Length", "Diameter", "Height", "Whole wt.", "Shucked wt.",
             "Viscera wt.", "Shell wt.", "Rings"]
    fig, ax = plt.subplots(figsize=(5.3, 4.05), layout="constrained")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("report", [CORAL, "#FFFFFF", TEAL])
    image = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(8), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(8), short, fontsize=8)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if matrix[i,j] > 0.75 else NAVY)
    fig.colorbar(image, ax=ax, shrink=0.74, label="Pearson r")
    save(fig, "abalone_correlations.pdf")

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.1, 2.7), layout="constrained")
    for ax, name, title in [(left, "Shell weight", "Shell weight"),
                             (right, "Shucked weight", "Meat weight")]:
        ax.scatter(frame[name], frame.Rings, s=7, color=TEAL, alpha=0.17,
                   linewidths=0, rasterized=True)
        ax.set(xlabel=name, ylabel="Rings", ylim=(0, 30),
               title=f"{title}: r = {frame[name].corr(frame.Rings):.2f}")
        ax.grid(axis="y")
    left.axvspan(0.2, 0.3, color=GOLD, alpha=0.16, zorder=0)
    assert int(frame["Shell weight"].between(0.2, 0.3, inclusive="left").sum()) == 1067
    save(fig, "abalone_scatter.pdf")

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.1, 2.75), layout="constrained")
    for ax, feature, title in [(left, "Shucked weight", "Meat weight"),
                                (right, "Shell weight", "Shell weight")]:
        for sex, color in [("F", CORAL), ("I", TEAL), ("M", NAVY)]:
            subset = frame.loc[frame.Sex == sex]
            bins = pd.qcut(subset[feature], 8, duplicates="drop")
            means = subset.groupby(bins, observed=True).agg(x=(feature, "mean"),
                rings=("Rings", "mean"))
            ax.plot(means.x, means.rings, "o-", color=color, markersize=3,
                    linewidth=1.4, label=sex)
        ax.set(xlabel=feature, ylabel="Mean rings", title=title)
        ax.grid(axis="y")
    right.legend(title="Sex", frameon=False, ncol=3, fontsize=8, title_fontsize=8, loc="upper left")
    save(fig, "abalone_trends.pdf")


if __name__ == "__main__":
    dia_figures()
    abalone_figures()
    print(f"Wrote twelve figures to {OUT}")

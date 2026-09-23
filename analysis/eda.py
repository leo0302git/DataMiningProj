"""Generate reproducible EDA tables and figures for DIA and Abalone."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "results" / "eda"
ABALONE_COLUMNS = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
    "Rings",
]


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def observed_type(series: pd.Series) -> str:
    """Classify a numeric feature by values observed in the current dataset."""
    if not pd.api.types.is_numeric_dtype(series) or series.isna().any():
        return "other"
    values = series.to_numpy()
    if not np.isfinite(values).all():
        return "other"
    unique = set(values)
    if len(unique) == 1:
        return "constant"
    if unique == {0, 1}:
        return "binary_0_1"
    if np.equal(values, np.floor(values)).all():
        return "integer"
    return "float"


def feature_summary(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    group: str | None = None,
    target: str | None = None,
) -> pd.DataFrame:
    numeric = frame[numeric_columns]
    summary = numeric.describe(percentiles=[0.25, 0.5, 0.75]).T.rename(
        columns={"25%": "q1", "50%": "median", "75%": "q3"}
    )
    summary.insert(0, "observed_type", [observed_type(numeric[column]) for column in numeric_columns])
    summary["mode"] = numeric.mode(dropna=True).iloc[0]
    summary["missing"] = numeric.isna().sum()
    summary["unique"] = numeric.nunique(dropna=True)
    summary["zero_fraction"] = numeric.eq(0).mean()
    summary["iqr"] = summary["q3"] - summary["q1"]
    summary["skew"] = numeric.skew()
    summary["max_abs_z"] = ((numeric - numeric.mean()) / numeric.std()).abs().max()
    low = summary["q1"] - 1.5 * summary["iqr"]
    high = summary["q3"] + 1.5 * summary["iqr"]
    summary["iqr_outliers"] = numeric.lt(low).sum() + numeric.gt(high).sum()

    if target is not None:
        variable = summary.index[summary["std"] > 0]
        summary[f"corr_{target}"] = np.nan
        summary.loc[variable, f"corr_{target}"] = numeric[variable].corrwith(frame[target])
    if group is not None:
        for value, subset in frame.groupby(group, observed=True):
            suffix = f"{group}_{value}".replace(" ", "_").lower()
            summary[f"mean_{suffix}"] = subset[numeric_columns].mean()
            summary[f"median_{suffix}"] = subset[numeric_columns].median()

    return summary


def heatmap(correlation: pd.DataFrame, title: str, path: Path) -> None:
    size = max(7, 0.45 * len(correlation))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(correlation)), correlation.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(correlation)), correlation.index, fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson r")
    save(fig, path)


def plot_grid(
    columns: list[str],
    draw,
    title: str,
    path: Path,
    ncols: int = 3,
) -> None:
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, column in zip(axes, columns, strict=False):
        draw(ax, column)
        ax.set_title(column)
    for ax in axes[len(columns) :]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    save(fig, path)


def load_dia() -> pd.DataFrame:
    files = [
        RAW / "dia" / "DIA_trainingset_RDKit_descriptors.csv",
        RAW / "dia" / "DIA_testset_RDKit_descriptors.csv",
    ]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("DIA files are missing; run download_data.py first")
    frame = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    if not {"Label", "SMILES"}.issubset(frame.columns):
        raise ValueError("DIA data must contain Label and SMILES")
    return frame


def run_dia() -> None:
    frame = load_dia()
    features = [c for c in frame.columns if c not in {"Label", "SMILES"}]
    output = OUTPUT / "dia"
    output.mkdir(parents=True, exist_ok=True)
    summary = feature_summary(frame, features, target="Label", group="Label")
    summary.to_csv(output / "feature_summary.csv", float_format="%.6g")

    counts = frame["Label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Negative (0)", "Positive (1)"], counts.values, color=["#4C78A8", "#E45756"])
    for index, count in enumerate(counts.values):
        ax.text(index, count, f"{count}\n({count / len(frame):.1%})", ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.16)
    ax.set_ylabel("Samples")
    ax.set_title("DIA class distribution")
    save(fig, output / "class_distribution.png")

    effect = (
        (summary["mean_label_1"] - summary["mean_label_0"]).abs()
        / summary["std"].replace(0, np.nan)
    )
    class_features = effect.nlargest(6).index.tolist()

    def dia_boxplot(ax: plt.Axes, column: str) -> None:
        ax.boxplot(
            [frame.loc[frame["Label"] == label, column].dropna() for label in [0, 1]],
            tick_labels=["Negative", "Positive"],
            showfliers=True,
        )
        ax.set_ylabel("Value")

    plot_grid(class_features, dia_boxplot, "DIA: features with largest class shifts", output / "boxplots.png")

    distribution_features = (
        summary.loc[(summary["unique"] > 10) & (summary["std"] > 0), "skew"]
        .abs()
        .nlargest(6)
        .index.tolist()
    )

    def dia_histogram(ax: plt.Axes, column: str) -> None:
        for label, color, name in [(0, "#4C78A8", "Negative"), (1, "#E45756", "Positive")]:
            ax.hist(
                frame.loc[frame["Label"] == label, column],
                bins=25,
                density=True,
                alpha=0.55,
                color=color,
                label=name,
            )
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    plot_grid(distribution_features, dia_histogram, "DIA: most skewed continuous descriptors", output / "histograms.png")

    def dia_quantile(ax: plt.Axes, column: str) -> None:
        for label, color, name in [(0, "#4C78A8", "Negative"), (1, "#E45756", "Positive")]:
            values = np.sort(frame.loc[frame["Label"] == label, column].to_numpy())
            ax.plot(np.linspace(0, 1, len(values)), values, color=color, label=name)
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Value")
        ax.legend(fontsize=7)

    plot_grid(distribution_features, dia_quantile, "DIA: quantile plots", output / "quantile_plots.png")

    correlation_features = summary[f"corr_Label"].abs().nlargest(15).index.tolist()
    heatmap(
        frame[correlation_features].corr(),
        "DIA: correlation among descriptors most associated with Label",
        output / "correlation.png",
    )

    variable_features = summary.index[summary["std"] > 0].tolist()
    correlation = frame[variable_features].corr().abs()
    pairs = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack()
    top_pairs = pairs.nlargest(3).index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (x, y) in zip(axes, top_pairs, strict=True):
        for label, color, name in [(0, "#4C78A8", "Negative"), (1, "#E45756", "Positive")]:
            subset = frame[frame["Label"] == label]
            ax.scatter(subset[x], subset[y], s=12, alpha=0.45, color=color, label=name)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"|r| = {correlation.loc[x, y]:.3f}")
    axes[0].legend(fontsize=8)
    fig.suptitle("DIA: most correlated descriptor pairs")
    fig.tight_layout()
    save(fig, output / "scatterplots.png")

    print(f"DIA: {len(frame)} rows, {len(features)} descriptors -> {output}")


def load_abalone() -> pd.DataFrame:
    path = RAW / "abalone" / "abalone.data"
    if not path.is_file():
        raise FileNotFoundError("Abalone file is missing; run download_data.py first")
    frame = pd.read_csv(path, header=None, names=ABALONE_COLUMNS)
    if set(frame["Sex"].unique()) != {"M", "F", "I"}:
        raise ValueError("Unexpected Abalone Sex values")
    return frame


def run_abalone() -> None:
    frame = load_abalone()
    numeric = ABALONE_COLUMNS[1:]
    features = numeric[:-1]
    output = OUTPUT / "abalone"
    output.mkdir(parents=True, exist_ok=True)
    summary = feature_summary(frame, numeric, target="Rings", group="Sex")
    summary.to_csv(output / "feature_summary.csv", float_format="%.6g")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sex_counts = frame["Sex"].value_counts().reindex(["M", "F", "I"])
    axes[0].bar(["Male", "Female", "Infant"], sex_counts, color=["#4C78A8", "#E45756", "#72B7B2"])
    axes[0].set_title("Sex distribution")
    axes[0].set_ylabel("Samples")
    axes[1].hist(frame["Rings"], bins=np.arange(0.5, 30.5, 1), color="#59A14F", edgecolor="white")
    axes[1].axvline(frame["Rings"].median(), color="black", linestyle="--", label="Median")
    axes[1].set_title("Rings distribution")
    axes[1].set_xlabel("Rings")
    axes[1].set_ylabel("Samples")
    axes[1].legend()
    fig.tight_layout()
    save(fig, output / "target_distribution.png")

    def abalone_boxplot(ax: plt.Axes, column: str) -> None:
        ax.boxplot(
            [frame.loc[frame["Sex"] == sex, column] for sex in ["M", "F", "I"]],
            tick_labels=["M", "F", "I"],
            showfliers=True,
        )
        ax.set_ylabel("Value")

    plot_grid(numeric, abalone_boxplot, "Abalone: distributions by Sex", output / "boxplots.png", ncols=4)

    def abalone_histogram(ax: plt.Axes, column: str) -> None:
        ax.hist(frame[column], bins=25, color="#4C78A8", edgecolor="white")
        ax.axvline(frame[column].median(), color="#E45756", linestyle="--", label="Median")
        ax.set_ylabel("Samples")
        ax.legend(fontsize=7)

    plot_grid(numeric, abalone_histogram, "Abalone: histograms", output / "histograms.png", ncols=4)

    def abalone_quantile(ax: plt.Axes, column: str) -> None:
        values = np.sort(frame[column].to_numpy())
        ax.plot(np.linspace(0, 1, len(values)), values, color="#4C78A8")
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Value")

    plot_grid(numeric, abalone_quantile, "Abalone: quantile plots", output / "quantile_plots.png", ncols=4)
    heatmap(frame[numeric].corr(), "Abalone: Pearson correlation", output / "correlation.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    colors = frame["Sex"].map({"M": "#4C78A8", "F": "#E45756", "I": "#72B7B2"})
    for ax, feature in zip(axes, features, strict=False):
        ax.scatter(frame[feature], frame["Rings"], c=colors, s=10, alpha=0.35)
        coefficients = np.polyfit(frame[feature], frame["Rings"], 1)
        x_line = np.linspace(frame[feature].min(), frame[feature].max(), 100)
        ax.plot(x_line, np.polyval(coefficients, x_line), color="black", linewidth=1)
        ax.set_xlabel(feature)
        ax.set_ylabel("Rings")
        ax.set_title(f"r = {frame[feature].corr(frame['Rings']):.3f}")
    for ax in axes[len(features) :]:
        ax.set_visible(False)
    fig.suptitle("Abalone: physical measurements vs Rings")
    fig.tight_layout()
    save(fig, output / "scatterplots.png")

    print(f"Abalone: {len(frame)} rows, {len(features)} numeric features -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "dia", "abalone"], default="all")
    args = parser.parse_args()
    if args.dataset in {"all", "dia"}:
        run_dia()
    if args.dataset in {"all", "abalone"}:
        run_abalone()


if __name__ == "__main__":
    main()

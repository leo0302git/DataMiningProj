"""Convert the official DIA CSV files to RRL's .data/.info format."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "dia"
OUTPUT_DIR = Path(__file__).resolve().parent / "dataset"


def build_dia_dataset(raw_dir=RAW_DIR, output_dir=OUTPUT_DIR):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    train = pd.read_csv(raw_dir / "DIA_trainingset_RDKit_descriptors.csv")
    test = pd.read_csv(raw_dir / "DIA_testset_RDKit_descriptors.csv")
    if list(train.columns) != list(test.columns):
        raise ValueError("DIA training and test columns do not match")

    data = pd.concat([train, test], ignore_index=True)
    feature_names = [name for name in data.columns if name not in {"Label", "SMILES"}]
    features = data[feature_names]

    if data.shape != (597, 198) or len(feature_names) != 196:
        raise ValueError(f"Unexpected DIA shape: {data.shape}, features={len(feature_names)}")
    if data["Label"].value_counts().sort_index().to_dict() != {0: 449, 1: 148}:
        raise ValueError("Unexpected DIA label distribution")
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("DIA descriptors contain missing or non-finite values")

    binary_features = [
        name for name in feature_names if set(features[name].unique()) == {0, 1}
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_data = data[feature_names + ["Label"]]
    output_data.to_csv(output_dir / "dia.data", index=False, header=False)

    binary_set = set(binary_features)
    info_lines = [
        f"{name} {'discrete' if name in binary_set else 'continuous'}"
        for name in feature_names
    ]
    info_lines.extend(["Label discrete", "LABEL_POS -1"])
    (output_dir / "dia.info").write_text("\n".join(info_lines) + "\n", encoding="utf-8")

    return {
        "samples": len(data),
        "features": len(feature_names),
        "binary_features": len(binary_features),
        "continuous_features": len(feature_names) - len(binary_features),
    }


if __name__ == "__main__":
    summary = build_dia_dataset()
    print(
        "Prepared DIA for RRL: "
        f"{summary['samples']} samples, {summary['features']} features "
        f"({summary['binary_features']} discrete, "
        f"{summary['continuous_features']} continuous)."
    )

from models.rrl.prepare_dia import build_dia_dataset


def test_prepare_dia(tmp_path):
    summary = build_dia_dataset(output_dir=tmp_path)

    assert summary == {
        "samples": 597,
        "features": 196,
        "binary_features": 20,
        "continuous_features": 176,
    }
    assert sum(1 for _ in (tmp_path / "dia.data").open()) == 597
    assert sum(1 for _ in (tmp_path / "dia.info").open()) == 198

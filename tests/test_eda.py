import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.eda import observed_type


def test_observed_type() -> None:
    assert observed_type(pd.Series([0, 0])) == "constant"
    assert observed_type(pd.Series([0, 1])) == "binary_0_1"
    assert observed_type(pd.Series([0, 2])) == "integer"
    assert observed_type(pd.Series([0.0, 0.5])) == "float"

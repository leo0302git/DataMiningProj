import numpy as np

from models.rrl.optimize_dia import candidate_configs, make_cut_points


def test_optimization_space_and_supervised_thresholds():
    configs = candidate_configs()
    assert len(configs) == 40
    assert {config['threshold'] for config in configs} == {
        'random', 'quantile', 'supervised'}
    assert {config['structure'] for config in configs} == {
        '3@16', '5@16', '5@32', '5@64', '10@32', '5@32@16'}

    X = np.linspace(-2, 2, 40).reshape(-1, 1)
    y = np.array([0] * 20 + [1] * 20)
    cut_points = make_cut_points(X, y, 3, 'supervised').numpy()
    assert cut_points.shape == (3, 1)
    assert np.isfinite(cut_points).all()
    assert np.any(np.abs(cut_points) < 0.5)

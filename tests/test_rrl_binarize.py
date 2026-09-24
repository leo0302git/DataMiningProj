import torch

from models.rrl.rrl.components import BinarizeLayer


def test_binarize_layer_accepts_fixed_cut_points():
    cut_points = torch.tensor([[0.0], [1.0]])
    layer = BinarizeLayer(2, (0, 1), cut_points=cut_points)
    output = layer(torch.tensor([[0.5]]))
    assert torch.equal(layer.cl, cut_points)
    assert torch.equal(output, torch.tensor([[1.0, 0.0, 0.0, 1.0]]))

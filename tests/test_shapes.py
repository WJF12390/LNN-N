import torch
from src.models.improved_lnn import ImprovedLNNRegressor, tabular_to_feature_sequence


def test_lnn_forward_shape():
    X = torch.randn(8, 5)
    X_seq = tabular_to_feature_sequence(X)
    model = ImprovedLNNRegressor(input_size=1, hidden_size=16, output_size=1)
    y = model(X_seq)
    assert y.shape == (8, 1)

import pytest
import torch

from ising_ood.models.factory import ModelFactory, count_parameters

IT_TIME = 32  # small for fast tests
BATCH = 3


@pytest.mark.parametrize("model_type", ["cnn", "improved_cnn", "gnn", "transformer", "hybrid"])
def test_forward_shape_is_66(model_type):
    kwargs = {"d_model": 16, "nhead": 2, "num_layers": 1} if model_type == "transformer" else {}
    if model_type == "gnn":
        kwargs = {"hidden_dim": 16}

    model = ModelFactory.create_model(model_type, IT_TIME, **kwargs)
    x = torch.randn(BATCH, IT_TIME, 12)

    out = model(x)
    assert out.shape == (BATCH, 66)
    assert count_parameters(model) > 0
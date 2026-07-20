import torch

from ising_ood.models.factory import output_to_tensor, inverse_tensor


def test_roundtrip_is_identity():
    torch.manual_seed(0)
    batch, dim = 4, 66
    v = torch.randint(0, 2, (batch, dim)).float()

    matrix = output_to_tensor(v)
    v_back = inverse_tensor(matrix)

    assert torch.allclose(v, v_back)


def test_matrix_is_symmetric_zero_diagonal():
    torch.manual_seed(1)
    v = torch.randint(0, 2, (2, 66)).float()
    matrix = output_to_tensor(v)

    assert torch.allclose(matrix, matrix.transpose(1, 2))
    assert torch.allclose(torch.diagonal(matrix, dim1=1, dim2=2), torch.zeros(2, 12))
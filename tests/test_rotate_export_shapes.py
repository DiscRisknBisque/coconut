import pytest
import torch

from preprocessing.train_rotate_pykeen import to_real_embedding


def test_to_real_embedding_from_complex_tensor():
    x = torch.randn(5, 4, dtype=torch.cfloat)
    out = to_real_embedding(x, expected_complex_dim=4)

    assert out.shape == (5, 8)
    assert out.dtype == torch.float32


def test_to_real_embedding_validates_real_dim():
    x = torch.randn(5, 7)
    with pytest.raises(ValueError):
        to_real_embedding(x, expected_complex_dim=4)

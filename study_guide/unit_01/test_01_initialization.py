import torch
import pytest

def test_q1_flatten_vs_view():
  """Q1: Verify difference between view and reshape on non-contiguous tensors."""
  x = torch.arange(12).reshape(3, 4)
  x_t = x.t()

  # Assert expected behavior directly
  assert not x_t.is_contiguous()
  with pytest.raises(RuntimeError):
    _ = x_t.view(-1)

  flattened = x_t.reshape(-1)
  assert flattened.shape == (12,)
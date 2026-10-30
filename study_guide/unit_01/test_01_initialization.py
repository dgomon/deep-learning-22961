import torch
import pytest

def test_q1_list():
  list_of_floats = [1.0, 2,0, 3.0]
  list_of_ints = [1, 2, 3]
  list_of_chars = ['a', 'b', 'c']

  tensor_of_floats = torch.tensor(list_of_floats, dtype=torch.float)
  tensor_of_ints = torch.tensor(list_of_ints, dtype=torch.int)
  with pytest.raises(AttributeError):
    torch.tensor(list_of_chars, dtype=torch.char)

  pass

def test_q1_four_dim_tensor():

  # Define 4-dimensional shape (e.g., batch_size, channels, height, width)
  shape = (2, 3, 4, 4)

  # torch.randint draws uniformly from [low, high), so high=2 produces 0 or 1
  tensor_bits = torch.randint(low=0, high=2, size=shape, dtype=torch.uint8)

  print(tensor_bits)
  print("Shape:", tensor_bits.shape)
  print("Dtype:", tensor_bits.dtype)

  pass
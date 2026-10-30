import torch

def test_q1():
  x = torch.ones(5).reshape(5, 1)
  address = id(x)
  print(x)

  x += x
  print(x)
  assert(address == id(x))
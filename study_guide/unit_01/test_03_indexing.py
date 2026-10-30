import torch

def test_q1():
  x = torch.randn((10, 10))
  y = torch.full((10, 10), torch.inf)
  z = torch.where(x > 0, x, y)
  print(z)
  pass

def test_q2():
  x = torch.randn((10, 10)).reshape(10, 5, 2)
  y = x[1::2, 1::2, 1::2]

  print(x)
  print(y)
  pass
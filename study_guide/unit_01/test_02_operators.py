import timeit

import torch
import pytest

def test_q1():
  x = torch.rand(100, 2, 2)

  eigenvals = torch.linalg.eig(x)
  # print(eigenvals)

  norm_per_matrix = torch.linalg.norm(x, ord="fro", dim=(-2, -1))
  # print(norm_per_matrix)

  euclidean_norm = torch.linalg.vector_norm(x)
  # print(euclidean_norm)

  norm_inf_0 = torch.linalg.norm(x[:, :, 0], ord=torch.inf)
  print(norm_inf_0)

  norm_inf_1 = torch.linalg.norm(x[:, :, 1], ord=torch.inf)
  print(norm_inf_1)

  pass

def test_q2():
  x = torch.randint(low = 0, high = 100, size = 10**5)
  y = torch.randint(low = 0, high = 100, size = 10**5)

  timeit
  x / y




  eigenvals = torch.linalg.eig(x)
  pass
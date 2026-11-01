import torch

def can_be_expanded(A, B):
  r"""
    Returns True iff A can be expanded into B
  """

  index_A = A.dim() - 1
  index_B = B.dim() - 1

  if index_A > index_B:
    # A has more dimensions than B
    return False

  while index_A >= 0:
    if A.shape[index_A] != 1 and A.shape[index_A] != B.shape[index_B]:
      # Current A's dimension can only be expanded if its length is 1 or its length matches that of B's
      return False

    index_A -= 1
    index_B -= 1
  return True

def my_expand_as(A, B):
  r"""
    Expands A into dimensions of B
  """
  if not can_be_expanded(A, B):
    raise RuntimeError

  # Clone to avoid in-place modification of A
  C = A.clone()

  # Add singleton dimensions at the beginning until ranks match
  while C.dim() < B.dim():
    C = C.unsqueeze(0)

  for dim in range(C.dim()):
    if C.shape[dim] == 1 and B.shape[dim] > 1:
      # Replicate along this axis by concatenating copies
      C = torch.cat([C] * B.shape[dim], dim=dim)

  return C

def can_be_broadcasted(A, B):
  r"""Return True iff A and Be can be broadcasted together"""

  for i in range(min(A.dim(), B.dim())):
    current_size_dim_A = A.shape[-1]
    current_size_dim_B = B.shape[-1]
    if (current_size_dim_A != current_size_dim_B) and (current_size_dim_A != 1) and (current_size_dim_B != 1):
      return False, None

  C = [1] * max(len(A.shape), len(B.shape))

  for i in range(min(len(A.shape), len(B.shape))):
    C[-i - 1] = max(A.size()[-i - 1], B.size()[-i - 1])

  if len(A.shape) > len(B.shape):
    for i in range(len(A.shape) - len(B.shape)):
      C[i] = A.shape[i]
  elif len(B.shape) > len(A.shape):
    for i in range(len(B.shape) - len(A.shape)):
      C[i] = B.shape[i]

  return True, torch.Size(C)

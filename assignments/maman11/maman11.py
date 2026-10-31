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
    size_A = A.shape[index_A]
    size_B = A.shape[index_B]
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
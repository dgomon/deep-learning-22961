import torch

def can_be_expanded(A: torch.Tensor, B: torch.Tensor) -> bool:
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

def my_expand_as(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
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

def can_be_broadcasted(A: torch.Tensor, B: torch.Tensor) -> tuple[bool, torch.Size | None]:
  r"""Determine if tensors A and B can be broadcasted together.

    Returns (True, broadcast_shape) if compatible, otherwise (False, None).
  """

  C = [0] * max(len(A.shape), len(B.shape))

  for i in range(1, min(A.dim(), B.dim()) + 1):
    a, b = A.shape[-i], B.shape[-i]
    if (a != b) and (a != 1) and (b != 1):
      return False, None
    # b may be 0 here, thus not using max()
    C[-i] = a if b == 1 else b

  diff = A.dim() - B.dim()
  if diff != 0:
      longer = A.shape if diff > 0 else B.shape
      C[:abs(diff)] = longer[:abs(diff)]

  return True, torch.Size(C)


def _broadcast_to(t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    res = t.clone()

    # Step 1: Align ranks by prepending singleton dimensions
    while res.dim() < len(target_shape):
        res = res.unsqueeze(0)

    # Step 2: Expand each dimension using torch.cat
    for d, target_size in enumerate(target_shape):
        curr_size = res.shape[d]
        if curr_size == 1 and target_size > 1:
            # Replicate along dimension d
            res = torch.cat([res] * target_size, dim=d)
        elif curr_size == 1 and target_size == 0:
            # Handle size 0 edge case: reshaping empty tensor to target slice
            empty_shape = list(res.shape)
            empty_shape[d] = 0
            res = torch.empty(empty_shape, dtype=res.dtype, device=res.device)
    return res

def my_broadcast(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    possible, size_C = can_be_broadcasted(A, B)

    if not possible:
        raise RuntimeError(f"The size of tensor A {A.shape} must match the size of tensor B {B.shape}")

    result_A = _broadcast_to(A, size_C)
    result_B = _broadcast_to(B, size_C)

    return result_A, result_B

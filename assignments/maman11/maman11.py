import torch


def _can_expand_to(source: torch.Tensor, target: torch.Tensor) -> bool:
    """Check whether a source tensor can be expanded to the shape of a target tensor.

    Follows PyTorch broadcasting semantics: the source tensor must have fewer
    than or equal dimensions compared to the target, and trailing dimensions
    must either match the target's size or equal 1.

    Args:
        source: Tensor whose shape is evaluated for expansion.
        target: Target tensor providing the desired output shape.

    Returns:
        bool: True if source.expand_as(target) is valid, False otherwise.
    """
    index_A = source.dim() - 1
    index_B = target.dim() - 1

    if index_A > index_B:
        # A has more dimensions than B
        return False

    while index_A >= 0:
        if source.shape[index_A] != 1 and source.shape[index_A] != target.shape[index_B]:
            # Current A's dimension can only be expanded if its length is 1 or its length matches that of B's
            return False

        index_A -= 1
        index_B -= 1
    return True


def my_expand_as(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    r"""Expand `input` to the same size as `other`.

        Equivalent to calling an explicit tensor expansion along singleton
        dimensions to match `other.size()`.

        Args:
            input (Tensor): the input tensor.
            other (Tensor): the tensor whose size should be matched.

        Returns:
            Tensor: A new tensor matching `other.shape`.

        Raises:
            RuntimeError: If `input` cannot be expanded to the size of `other`.
        """
    if not _can_expand_to(input, other):
        raise RuntimeError

    # Clone to avoid in-place modification of input
    c = input.clone()

    # Add singleton dimensions at the beginning until ranks match
    while c.dim() < other.dim():
        c = c.unsqueeze(0)

    for dim in range(c.dim()):
        if c.shape[dim] == 1 and other.shape[dim] > 1:
            # Replicate along this axis by concatenating copies
            c = torch.cat([c] * other.shape[dim], dim=dim)

    return c


def can_be_broadcasted(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, torch.Size | None]:
    r"""Checks whether two tensors are broadcastable according to PyTorch rules.

        Args:
            a: First input tensor.
            b: Second input tensor.

        Returns:
            A tuple `(is_broadcastable, broadcast_shape)`:
            - `is_broadcastable`: `True` if tensors can be broadcast together, `False` otherwise.
            - `broadcast_shape`: The resulting `torch.Size` if broadcastable, or `None`.
        """

    c = [0] * max(len(a.shape), len(b.shape))

    for i in range(1, min(a.dim(), b.dim()) + 1):
        a, b = a.shape[-i], b.shape[-i]
        if (a != b) and (a != 1) and (b != 1):
            return False, None
        # b may be 0 here, thus not using max()
        c[-i] = a if b == 1 else b

    diff = a.dim() - b.dim()
    if diff != 0:
        longer = a.shape if diff > 0 else b.shape
        c[:abs(diff)] = longer[:abs(diff)]

    return True, torch.Size(c)


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

import torch
from sympy import resultant


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


def my_expand_as(input_tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    r"""Expand `input` to the same size as `other`.

        Equivalent to calling an explicit tensor expansion along singleton
        dimensions to match `other.size()`.

        Args:
            input_tensor (Tensor): the input tensor.
            other (Tensor): the tensor whose size should be matched.

        Returns:
            Tensor: A new tensor matching `other.shape`.

        Raises:
            RuntimeError: If `input_tensor` cannot be expanded to the size of `other`.
        """
    if not _can_expand_to(input_tensor, other):
        raise RuntimeError

    # Clone to avoid in-place modification of input
    c = input_tensor.clone()

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
        dim_a, dim_b = a.shape[-i], b.shape[-i]
        if (dim_a != dim_b) and (dim_a != 1) and (dim_b != 1):
            return False, None
        # b may be 0 here, thus not using max()
        c[-i] = dim_a if dim_b == 1 else dim_b

    diff = a.dim() - b.dim()
    if diff != 0:
        longer = a.shape if diff > 0 else b.shape
        c[:abs(diff)] = longer[:abs(diff)]

    return True, torch.Size(c)


def _broadcast_to(t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Explicitly broadcast a tensor to a target shape using repetition.

        Aligns tensor rank by prepending singleton dimensions, then replicates
        singleton dimensions via concatenation to match the specified shape.

        Args:
            t: Input tensor to broadcast.
            target_shape: Destination shape. Each dimension must be compatible
                with broadcasting rules (equal to the current dimension or current is 1).

        Returns:
            A new tensor broadcasted to `target_shape`.
        """
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


def my_broadcast(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast two tensors to a common compatible shape.

        Args:
            a: First input tensor.
            b: Second input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing both tensors
            broadcasted to the shared target shape.

        Raises:
            RuntimeError: If shapes of `a` and `b` are incompatible for broadcasting.
        """
    possible, size_c = can_be_broadcasted(a, b)

    if not possible:
        raise RuntimeError(f"The size of tensor A {a.shape} must match the size of tensor B {b.shape}")

    result_a = _broadcast_to(a, size_c)
    result_b = _broadcast_to(b, size_c)

    return result_a, result_b

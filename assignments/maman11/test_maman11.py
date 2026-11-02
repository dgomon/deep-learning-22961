import pytest
import torch

from assignments.maman11.maman11 import can_be_expanded, my_expand_as, can_be_broadcasted, my_broadcast


def test_can_be_expanded():
  data = (
    ((1,), (1,), True),
    ((1,), (5,), True),
    ((5,), (1,), False),
    ((2,), (5, 2,), True),
    ((1,), (5, 2,), True),
    ((3,), (5, 2,), False),
    ((1, 2), (5, 2,), True),
    ((5, 2), (5, 2,), True),
    ((1, 3), (3, 1,), False),
    ((1, 3), (3, 3,), True),
    ((5, 2), (8, 5, 2,), True),
  )

  for entry in data:
    A = torch.empty(entry[0])
    B = torch.empty(entry[1])

    assert can_be_expanded(A, B) == entry[2]


def test_expand():
  data = (
    (
      torch.tensor([1]),
      torch.tensor([3, 2]),
      torch.tensor([1, 1],)
    ),
    (
      torch.tensor([5, 7]),
      torch.tensor([3, 2]),
      torch.tensor([5, 7],)
    ),
    (
      torch.tensor([5, 7]),
      torch.tensor([[1, 2],
                    [3, 4]]),
      torch.tensor([[5, 7],
                    [5, 7]]),
    ),
    (
      torch.tensor([1]),
      torch.tensor([[3, 2],
                    [4, 6]]),
      torch.tensor([[1, 1],
                    [1, 1]]),
    ),
    (
      torch.tensor([97, 17, 2]),
      torch.tensor([[11, 12, 13],
                   [21, 22, 23],
                   [31, 32, 33]]),
      torch.tensor([[97, 17,  2],
                    [97, 17,  2],
                    [97, 17,  2]])
    ),
  )

  for entry in data:
    torch.testing.assert_close(my_expand_as(entry[0], entry[1]), entry[2])


def test_can_be_broadcasted():
  data = (
    # --- 1. Scalars / 0-D Tensors () ---
    # Broadcast with scalar should retain the other shape unchanged
    (((), ()), (True, ())),
    (((), (1,)), (True, (1,))),
    (((3, 4), ()), (True, (3, 4))),
    (((), (2, 5, 1)), (True, (2, 5, 1))),

    # --- 2. Zero-sized / Empty dimensions (size 0) ---
    # 0 can broadcast with 1 or 0, but not with n > 1
    (((0,), (0,)), (True, (0,))),
    (((0,), (1,)), (True, (0,))),
    (((1,), (0,)), (True, (0,))),
    (((0,), (3,)), (False, None)),
    (((2, 0, 1), (1, 1, 4)), (True, (2, 0, 4))),
    (((2, 0, 3), (2, 1, 4)), (False, None)),

    # # --- 3. Mismatches at non-trailing positions ---
    # # Trailing match, mismatch occurs at inner dimensions
    (((2, 3), (4, 3)), (False, None)),
    (((2, 5, 4), (3, 5, 4)), (False, None)),
    (((2, 1, 4), (3, 5, 4)), (False, None)),
    (((5, 1, 2), (5, 3, 4)), (False, None)),
    #
    # # --- 4. Alternating / Cross-broadcasting (both tensors expand) ---
    # # Multiple alternating singleton dimensions across higher ranks
    (((1, 4, 1, 6), (3, 1, 5, 1)), (True, (3, 4, 5, 6))),
    (((2, 1, 2, 1), (1, 3, 1, 3)), (True, (2, 3, 2, 3))),
    #
    # # --- 5. High rank difference ---
    # # Significant dimensionality gap (e.g., 1D vs 5D)
    (((4,), (2, 1, 3, 1, 4)), (True, (2, 1, 3, 1, 4))),
    (((1,), (2, 3, 4, 5, 6)), (True, (2, 3, 4, 5, 6))),
    (((5,), (2, 3, 4, 5, 6)), (False, None)),
    # #
    # # # --- 6. All ones (unit tensors) ---
    (((1, 1, 1), (1, 1, 1)), (True, (1, 1, 1))),
    (((1, 1, 1), (4, 5)), (True, (1, 4, 5))),
  )

  for (shape_a, shape_b), (expected_broadcastable, expected_shape) in data:
      A = torch.empty(shape_a)
      B = torch.empty(shape_b)

      status, shape = can_be_broadcasted(A, B)
      msg = f"Failed for shapes {shape_a} and {shape_b}"

      assert status == expected_broadcastable, f"{msg}: expected status {expected_broadcastable}, got {status}"
      assert shape == expected_shape, f"{msg}: expected shape {expected_shape}, got {shape}"
  pass


def test_my_broadcast_valid_cases():
    test_cases = [
        # 1. 1D scalar-like tensor with 2D tensor
        (
            torch.tensor([1]),
            torch.tensor([[11, 12, 13], [21, 22, 23]]),
            torch.tensor([[1, 1, 1], [1, 1, 1]]),
            torch.tensor([[11, 12, 13], [21, 22, 23]]),
        ),
        # 2. 0D scalar tensors
        (
            torch.tensor(5),
            torch.tensor([1, 2, 3]),
            torch.tensor([5, 5, 5]),
            torch.tensor([1, 2, 3]),
        ),
        # 3. Two 0D scalar tensors
        (
            torch.tensor(7),
            torch.tensor(8),
            torch.tensor(7),
            torch.tensor(8),
        ),
        # 4. Identical shapes (no expansion needed)
        (
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[5, 6], [7, 8]]),
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[5, 6], [7, 8]]),
        ),
        # 5. Row vector (1, N) with Column vector (M, 1) -> (M, N)
        (
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4], [5]]),
            torch.tensor([[1, 2, 3], [1, 2, 3]]),
            torch.tensor([[4, 4, 4], [5, 5, 5]]),
        ),
        # 6. Rank extension: 1D (3,) with 2D (2, 3) -> prepend dimension to (1, 3) then broadcast
        (
            torch.tensor([10, 20, 30]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[10, 20, 30], [10, 20, 30]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
        ),
        # 7. Higher dimension expansion: (2, 1, 3) with (1, 4, 1) -> (2, 4, 3)
        (
            torch.arange(6).reshape(2, 1, 3),
            torch.arange(4).reshape(1, 4, 1),
            torch.arange(6).reshape(2, 1, 3).expand(2, 4, 3),
            torch.arange(4).reshape(1, 4, 1).expand(2, 4, 3),
        ),
        # 8. Empty tensor with 0-sized dimension: (0, 1) with (3, 1) -> (0, 1)
        (
            torch.empty((0, 1)),
            torch.tensor([[1], [2], [3]]),
            torch.empty((0, 1)),
            torch.empty((0, 1)),
        ),
    ]

    for A, B, expected_A, expected_B in test_cases:
        actual_A, actual_B = my_broadcast(A, B)

        # Validate values and shapes match expectations
        torch.testing.assert_close(actual_A, expected_A)
        torch.testing.assert_close(actual_B, expected_B)

        # Validate consistency with native PyTorch broadcasting logic
        torch_exp_A, torch_exp_B = torch.broadcast_tensors(A, B)
        torch.testing.assert_close(actual_A, torch_exp_A)
        torch.testing.assert_close(actual_B, torch_exp_B)


def test_my_broadcast_invalid_cases():
    invalid_pairs = [
        # Incompatible trailing dimensions: (2,) vs (3,)
        (torch.tensor([1, 2]), torch.tensor([1, 2, 3])),
        # Incompatible inner dimensions: (2, 3) vs (2, 4)
        (torch.zeros(2, 3), torch.zeros(2, 4)),
        # Incompatible leading dimensions after right-alignment: (3, 2) vs (4, 2)
        (torch.zeros(3, 2), torch.zeros(4, 2)),
        # Multi-dimensional mismatch: (2, 3, 4) vs (2, 5, 4)
        (torch.zeros(2, 3, 4), torch.zeros(2, 5, 4)),
    ]

    for A, B in invalid_pairs:
        with pytest.raises(RuntimeError):
            my_broadcast(A, B)
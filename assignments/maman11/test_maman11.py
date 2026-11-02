import pytest
import torch

from assignments.maman11.maman11 import can_be_expanded, my_expand_as, can_be_broadcasted, my_broadcast


VALID_EXPAND_CASES = [
    # (input_tensor, target_tensor, expected_output, case_id)
    (
        torch.tensor([1]),
        torch.tensor([3, 2]),
        "1d_singleton_to_1d",
    ),
    (
        torch.tensor([5, 7]),
        torch.tensor([3, 2]),
        "1d_same_shape",
    ),
    (
        torch.tensor([5, 7]),
        torch.tensor([[1, 2], [3, 4]]),
        "1d_to_2d_row_broadcast",
    ),
    (
        torch.tensor([1]),
        torch.tensor([[3, 2], [4, 6]]),
        "1d_singleton_to_2d",
    ),
    (
        torch.tensor([97, 17, 2]),
        torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]]),
        "1d_to_square_2d",
    ),
    # --- Augmented Cases ---
    (
        torch.tensor(42),
        torch.tensor([[1, 2], [3, 4]]),
        "0d_scalar_to_2d",
    ),
    (
        torch.tensor([[1], [2]]),
        torch.tensor([[0, 0, 0], [0, 0, 0]]),
        "2d_column_singleton_to_2d",
    ),
    (
        torch.tensor([1, 2]),
        torch.empty(2, 2, 2),
        "1d_to_3d_expansion",
    ),
]

INVALID_EXPAND_CASES = [
    # Incompatible trailing dimension (2 != 3)
    (
        torch.tensor([1, 2]),
        torch.tensor([1, 2, 3]),
        "mismatched_trailing_dim",
    ),
    # Attempting to expand a non-singleton dimension (2 -> 3)
    (
        torch.tensor([[1, 2], [3, 4]]),
        torch.empty(3, 2),
        "incompatible_leading_dim",
    ),
    # Input has more dimensions than target
    (
        torch.tensor([[1, 2]]),
        torch.tensor([1, 2]),
        "input_higher_dim_than_target",
    ),
]

ALL_EXPAND_CASES = [
    (*case[:2], True, case[2]) for case in VALID_EXPAND_CASES
] + [
    (*case[:2], False, case[2]) for case in INVALID_EXPAND_CASES
]

@pytest.mark.parametrize(
    "source, target",
    [case[:2] for case in ALL_EXPAND_CASES],
    ids=[case[2] for case in ALL_EXPAND_CASES],
)
def test_can_be_expanded(
    source: torch.Tensor, target: torch.Tensor,
) -> None:
  actual = can_be_expanded(source, target)
  if actual:
      # Must NOT raise an exception when expandable
      try:
          my_expand_as(source, target)
      except RuntimeError as exc:
          pytest.fail(f"can_be_expanded returned True, but my_expand_as raised: {exc}")
  else:
      # MUST raise RuntimeError when not expandable
      with pytest.raises(RuntimeError):
          my_expand_as(source, target)


@pytest.mark.parametrize(
    "source, target",
    [case[:2] for case in VALID_EXPAND_CASES],
    ids=[case[2] for case in VALID_EXPAND_CASES],
)
def test_my_expand_as_valid(source: torch.Tensor, target: torch.Tensor):
    result = my_expand_as(source, target)

    assert result.shape == target.shape
    orig_expected = source.expand(target.shape)
    torch.testing.assert_close(result, orig_expected)


@pytest.mark.parametrize(
    "source, target",
    [case[:2] for case in INVALID_EXPAND_CASES],
    ids=[case[2] for case in INVALID_EXPAND_CASES],
)
def test_my_expand_as_incompatible_shapes(source: torch.Tensor, target: torch.Tensor):
    with pytest.raises(RuntimeError):
      my_expand_as(source, target)


def get_expected_broadcast(shape_a, shape_b):
    try:
        expected_shape = tuple(torch.broadcast_shapes(shape_a, shape_b))
        return True, expected_shape
    except RuntimeError:
        return False, None


def test_can_be_broadcasted():
    test_shapes = (
        # --- 1. Scalars / 0-D Tensors () ---
        ((), ()),
        ((), (1,)),
        ((3, 4), ()),
        ((), (2, 5, 1)),
        # --- 2. Zero-sized / Empty dimensions (size 0) ---
        ((0,), (0,)),
        ((0,), (1,)),
        ((1,), (0,)),
        ((0,), (3,)),
        ((2, 0, 1), (1, 1, 4)),
        ((2, 0, 3), (2, 1, 4)),
        # --- 3. Mismatches at non-trailing positions ---
        ((2, 3), (4, 3)),
        ((2, 5, 4), (3, 5, 4)),
        ((2, 1, 4), (3, 5, 4)),
        ((5, 1, 2), (5, 3, 4)),
        # --- 4. Alternating / Cross-broadcasting ---
        ((1, 4, 1, 6), (3, 1, 5, 1)),
        ((2, 1, 2, 1), (1, 3, 1, 3)),
        # --- 5. High rank difference ---
        ((4,), (2, 1, 3, 1, 4)),
        ((1,), (2, 3, 4, 5, 6)),
        ((5,), (2, 3, 4, 5, 6)),
        # --- 6. All ones (unit tensors) ---
        ((1, 1, 1), (1, 1, 1)),
        ((1, 1, 1), (4, 5)),
    )

    for shape_a, shape_b in test_shapes:
        expected_broadcastable, expected_shape = get_expected_broadcast(shape_a, shape_b)

        A = torch.empty(shape_a)
        B = torch.empty(shape_b)

        status, shape = can_be_broadcasted(A, B)
        msg = f"Failed for shapes {shape_a} and {shape_b}"

        assert (
            status == expected_broadcastable
        ), f"{msg}: expected status {expected_broadcastable}, got {status}"
        assert (
            shape == expected_shape
        ), f"{msg}: expected shape {expected_shape}, got {shape}"


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
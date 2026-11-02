from typing import Tuple

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


TEST_SHAPES = (
    # 1. Scalars / 0-D Tensors
    pytest.param((), (), id="scalar_with_scalar"),
    pytest.param((), (1,), id="scalar_with_1d_unit"),
    pytest.param((3, 4), (), id="2d_with_scalar"),
    pytest.param((), (2, 5, 1), id="scalar_with_3d"),
    # 2. Zero-sized / Empty dimensions (size 0)
    pytest.param((0,), (0,), id="zero_with_zero"),
    pytest.param((0,), (1,), id="zero_with_unit"),
    pytest.param((1,), (0,), id="unit_with_zero"),
    pytest.param((0,), (3,), id="zero_with_incompatible_dim"),
    pytest.param((2, 0, 1), (1, 1, 4), id="zero_dim_broadcast_compatible"),
    pytest.param((2, 0, 3), (2, 1, 4), id="zero_dim_broadcast_incompatible"),
    # 3. Mismatches at non-trailing positions
    pytest.param((2, 3), (4, 3), id="leading_dim_mismatch_2d"),
    pytest.param((2, 5, 4), (3, 5, 4), id="leading_dim_mismatch_3d"),
    pytest.param((2, 1, 4), (3, 5, 4), id="leading_dim_mismatch_with_unit"),
    pytest.param((5, 1, 2), (5, 3, 4), id="trailing_dim_mismatch"),
    # 4. Alternating / Cross-broadcasting
    pytest.param((1, 4, 1, 6), (3, 1, 5, 1), id="cross_broadcast_4d_valid"),
    pytest.param((2, 1, 2, 1), (1, 3, 1, 3), id="cross_broadcast_4d_alternating"),
    # 5. High rank difference
    pytest.param((4,), (2, 1, 3, 1, 4), id="1d_to_5d_match"),
    pytest.param((1,), (2, 3, 4, 5, 6), id="1d_unit_to_5d"),
    pytest.param((5,), (2, 3, 4, 5, 6), id="1d_mismatch_trailing_5d"),
    # 6. All ones (unit tensors)
    pytest.param((1, 1, 1), (1, 1, 1), id="all_ones_same_rank"),
    pytest.param((1, 1, 1), (4, 5), id="all_ones_different_rank"),
)


@pytest.mark.parametrize("shape_a, shape_b", TEST_SHAPES)
def test_can_be_broadcasted(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> None:
    expected_status, expected_shape = get_expected_broadcast(shape_a, shape_b)

    # torch.empty on meta device avoids physical memory allocation
    tensor_a = torch.empty(shape_a, device="meta")
    tensor_b = torch.empty(shape_b, device="meta")

    status, shape = can_be_broadcasted(tensor_a, tensor_b)

    assert status == expected_status
    assert shape == expected_shape


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
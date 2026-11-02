import pytest
import torch

from assignments.maman11.maman11 import _can_be_expanded, my_expand_as, can_be_broadcasted, my_broadcast


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
  actual = _can_be_expanded(source, target)
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


VALID_BROADCAST_CASES = [
  (
      torch.tensor([1]),
      torch.tensor([[11, 12, 13], [21, 22, 23]]),
      "1d_scalar_like_with_2d",
  ),
  (
      torch.tensor(5),
      torch.tensor([1, 2, 3]),
      "0d_scalar_with_1d",
  ),
  (
      torch.tensor(7),
      torch.tensor(8),
      "two_0d_scalars",
  ),
  (
      torch.tensor([[1, 2], [3, 4]]),
      torch.tensor([[5, 6], [7, 8]]),
      "identical_shapes",
  ),
  (
      torch.tensor([[1, 2, 3]]),
      torch.tensor([[4], [5]]),
      "row_and_column_vectors",
  ),
  (
      torch.tensor([10, 20, 30]),
      torch.tensor([[1, 2, 3], [4, 5, 6]]),
      "rank_extension_1d_with_2d",
  ),
  (
      torch.arange(6).reshape(2, 1, 3),
      torch.arange(4).reshape(1, 4, 1),
      "higher_dimensional_expansion",
  ),
]

INVALID_BROADCAST_CASES = [
  (
      torch.tensor([1, 2]),
      torch.tensor([1, 2, 3]),
      "incompatible_trailing_dims",
  ),
  (
      torch.zeros(2, 3),
      torch.zeros(2, 4),
      "incompatible_inner_dims",
  ),
  (
      torch.zeros(3, 2),
      torch.zeros(4, 2),
      "incompatible_leading_dims",
  ),
  (
      torch.zeros(2, 3, 4),
      torch.zeros(2, 5, 4),
      "multidimensional_mismatch",
  ),
]

ALL_BROADCAST_CASES = [
    (*case[:2], True, case[2]) for case in VALID_BROADCAST_CASES
] + [
    (*case[:2], False, case[2]) for case in INVALID_BROADCAST_CASES
]

@pytest.mark.parametrize(
    "a, b",
    [case[:2] for case in ALL_BROADCAST_CASES],
    ids=[case[2] for case in ALL_BROADCAST_CASES],
)
def test_can_be_broadcasted(
        a: torch.Tensor, b: torch.Tensor
) -> None:
  (actual_are_compatible, actual_shape) = can_be_broadcasted(a, b)
  if actual_are_compatible:
      # Must NOT raise an exception when can be broadcasted
      try:
          expected = torch.broadcast_tensors(a, b)
      except RuntimeError as exc:
          pytest.fail(f"can_be_broadcasted returned True, but torch.broadcast_tensors raised: {exc}")
  else:
      # MUST raise RuntimeError when not expandable
      with pytest.raises(RuntimeError):
        torch.broadcast_tensors(a, b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param(
            torch.tensor([1]),
            torch.tensor([[11, 12, 13], [21, 22, 23]]),
            id="1d_scalar_like_with_2d",
        ),
        pytest.param(
            torch.tensor(5),
            torch.tensor([1, 2, 3]),
            id="0d_scalar_with_1d",
        ),
        pytest.param(
            torch.tensor(7),
            torch.tensor(8),
            id="two_0d_scalars",
        ),
        pytest.param(
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[5, 6], [7, 8]]),
            id="identical_shapes",
        ),
        pytest.param(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4], [5]]),
            id="row_and_column_vectors",
        ),
        pytest.param(
            torch.tensor([10, 20, 30]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            id="rank_extension_1d_with_2d",
        ),
        pytest.param(
            torch.arange(6).reshape(2, 1, 3),
            torch.arange(4).reshape(1, 4, 1),
            id="higher_dimensional_expansion",
        ),
    ],
)
def test_my_broadcast_valid_cases(a: torch.Tensor, b: torch.Tensor) -> None:
    expected_a, expected_b = torch.broadcast_tensors(a, b)
    actual_a, actual_b = my_broadcast(a, b)

    torch.testing.assert_close(actual_a, expected_a)
    torch.testing.assert_close(actual_b, expected_b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param(
            torch.tensor([1, 2]),
            torch.tensor([1, 2, 3]),
            id="incompatible_trailing_dims",
        ),
        pytest.param(
            torch.zeros(2, 3),
            torch.zeros(2, 4),
            id="incompatible_inner_dims",
        ),
        pytest.param(
            torch.zeros(3, 2),
            torch.zeros(4, 2),
            id="incompatible_leading_dims",
        ),
        pytest.param(
            torch.zeros(2, 3, 4),
            torch.zeros(2, 5, 4),
            id="multidimensional_mismatch",
        ),
    ],
)
def test_my_broadcast_invalid_cases(a: torch.Tensor, b: torch.Tensor) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        my_broadcast(a, b)
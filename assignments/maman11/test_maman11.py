import torch

from assignments.maman11.maman11 import can_be_expanded, my_expand_as, can_be_broadcasted


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
    (((1,), (1,)), (True, (1,))),
    (((1,), (5,)), (True, (5,))),
    (((5,), (1,)), (True, (5,))),
    (((2,), (2, 2,)), (True, (2, 2))),
    (((1,), (5, 2,)), (True, (5, 2))),
    (((3,), (5, 2,)), (False, None)),
    (((1, 2), (5, 2,)), (True, (5, 2))),
    (((5, 2), (5, 2,)), (True, (5, 2))),
    (((1, 3), (3, 1,)), (True, (3, 3))),
    (((1, 3), (3, 3,)), (True, (3, 3))),
    (((5, 2), (8, 5, 2,)), (True, (8, 5, 2))),
    (((8, 5, 2), (5, 2,)), (True, (8, 5, 2))),
    (((8, 5, 2), (5, 2, 3)), (False, None)),
  )

  for (shape_a, shape_b), (expected_broadcastable, expected_shape) in data:
      A = torch.empty(shape_a)
      B = torch.empty(shape_b)

      status, shape = can_be_broadcasted(A, B)
      msg = f"Failed for shapes {shape_a} and {shape_b}"

      assert status == expected_broadcastable, f"{msg}: expected status {expected_broadcastable}, got {status}"
      assert shape == expected_shape, f"{msg}: expected shape {expected_shape}, got {shape}"
  pass
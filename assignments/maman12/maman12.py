import torch


def my_sampler(size, dist, requires_grad = False):
    # Sum of probabilities must be one and each element must be positive
    assert (dist <= 0).sum() > 0 and dist.sum() == 1

    # Initialize result tensor to random values from U[0, 1]
    result = torch.rand(size, requires_grad = requires_grad)

    # Allocate mask of boolean to track assignments to result
    assigned = torch.empty_like(result, dtype = torch.bool)

    acc = 0.0 # Probability sum accumulator
    for i in range(len(dist)):
        acc += dist[i]
        # Mark entries whose accumulated probability exceeds threshold and that haven't been assigned yet
        mask = (acc > result) & (~assigned)
        result[mask] = i
        assigned[mask] = True

    return result


def test_b():
    n_repetitions = 10_000


def main():
    result = foo()
    print(result)


if __name__ == "__main__":
  main()

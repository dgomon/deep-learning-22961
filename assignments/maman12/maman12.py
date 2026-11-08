from matplotlib import pyplot as plt
import torch


def my_sampler(size, dist, requires_grad = False):
    # Convert dist list to tensor to parallelize calculations
    dist_tensor = torch.tensor(dist)

    # Sum of probabilities must be one and each element must be positive
    assert dist_tensor.sum() == 1 and (dist_tensor > 0).all()

    # Initialize result tensor to random values from U[0, 1]
    result = torch.rand(size, requires_grad = requires_grad)

    # Allocate mask of boolean to track assignments to result
    assigned = torch.zeros_like(result, dtype = torch.bool)

    cdf = 0.0 # Probability sum accumulator
    for i in range(len(dist_tensor)):
        cdf += dist_tensor[i]
        # Mark entries whose accumulated probability exceeds threshold and that haven't been assigned yet
        mask = (cdf > result) & (~assigned)
        result[mask] = i
        assigned[mask] = True

    return result


def run_b():
    size = 10_000
    dist = torch.tensor([0.1, 0.2, 0.7])
    result = my_sampler(size, [0.1, 0.2, 0.7], requires_grad = False)

    fig, axs = plt.subplots(1, 1, tight_layout=True)
    axs.hist(result, bins=len(dist))

    plt.show()
    pass

def main():
    run_b()
    pass

if __name__ == "__main__":
  main()

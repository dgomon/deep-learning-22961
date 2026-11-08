import torch


# def my_sampler(size, dist, requires_grad = False):

def verify_input(t: torch.Tensor) -> bool:
    # every element is positive
    if (t <= 0).sum() > 0 or t.sum() != 1:
        return False

    return True

def foo():
    t = torch.rand(20)
    acc = 0.0
    assigned = torch.empty_like(t, dtype = torch.bool)
    dist = (0.33, 0.33, 0.34)
    for i in range(len(dist)):
        acc += dist[i]
        mask = (acc > t) & (~assigned)
        t[mask] = i
        assigned[mask] = True
    return t


def main():
    result = foo()
    print(result)


if __name__ == "__main__":
  main()

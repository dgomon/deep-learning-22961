import torch


# def my_sampler(size, dist, requires_grad = False):


def foo():
    t = torch.rand(1)
    dist = (0.33, 0.33, 0.34)
    acc = 0
    assigned = False
    for i in range(len(dist)):
        acc += dist[i]
        if acc > t and not assigned:
            assigned = True
            t[0] = i
    if not assigned:
        t[0] = len(dist) - 1
    return t[0].item()


def main():
    result = []
    for i in range(100):
        result.append(foo())
    print("number of 0: ", result.count(0))
    print("number of 1: ", result.count(1))
    print("number of 2: ", result.count(2))
    # print(result)


if __name__ == "__main__":
  main()

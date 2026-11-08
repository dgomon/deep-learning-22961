from __future__ import annotations

import math

class MyScalar:
    def __init__(self, value, grad, parent):
        self.value = value
        self.grad = grad
        self.parent = parent

def plus(a: MyScalar, n: MyScalar) -> MyScalar:
    return MyScalar(a.value + n.value, 1, a)

def mul(a: MyScalar, n: MyScalar) -> MyScalar:
    return MyScalar(a.value * n.value, n.value, a)

def pow(a: MyScalar, n: MyScalar) -> MyScalar:
    return MyScalar(a.value ** n.value, n.value * a.value ** (n.value - 1), a)

def cos(a: MyScalar) -> MyScalar:
    return MyScalar(cos(a.value), -1 * math.sin(a.value), a)

def sin(a: MyScalar) -> MyScalar:
    return MyScalar(sin(a.value), math.cos(a.value), a)

def exp(a: MyScalar) -> MyScalar:
    return MyScalar(math.exp(a.value), math.exp(a.value), a)

def main():
    pass

def get_gradient(a: MyScalar) -> dict:
    if a.parent is None:
        return {a: 1}

    x = get_gradient(a.parent)
    x[a] = a.value * x[a.parent]
    return x


if __name__ == "__main__":
  main()

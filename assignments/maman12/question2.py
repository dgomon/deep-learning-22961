from __future__ import annotations

import math

class MyScalar:
    def __init__(self, value, grad, parent):
        self.value = value
        self.grad = grad
        self.parent = parent

def plus(a, n):
    return MyScalar(a.value + n.value, 1, a)

def mul(a, n):
    return MyScalar(a.value * n.value, n.value, a)

def power(a, n):
    return MyScalar(a.value ** n, n * a.value ** (n - 1), a)

def cos(a):
    return MyScalar(cos(a.value), -1 * math.sin(a.value), a)

def sin(a):
    return MyScalar(sin(a.value), math.cos(a.value), a)

def exp(a):
    return MyScalar(math.exp(a.value), math.exp(a.value), a)

def get_gradient(a) -> dict:
    if a.parent is None:
        return {a: 1}

    x = get_gradient(a.parent)
    x[a] = a.value * x[a.parent]
    return x

def main():
    a = MyScalar(2, 1, None)
    b = power(a,2) #a^2
    c = exp(b)
    d = get_gradient(c)
    print(d)
    pass

if __name__ == "__main__":
  main()

from __future__ import annotations

import math

import torch


class MyScalar:
    def __init__(self, value, grad, parent):
        self.value = value
        self.grad = grad
        self.parent = parent

def plus(a, n):
    return MyScalar(a.value + n, 1.0, a)

def mul(a, n):
    return MyScalar(a.value * n, n, a)

def pow(a, n):
    return MyScalar(a.value ** n, n * a.value ** (n - 1), a)

def cos(a):
    return MyScalar(cos(a.value), -1 * math.sin(a.value), a)

def sin(a):
    return MyScalar(sin(a.value), math.cos(a.value), a)

def exp(a):
    return MyScalar(math.exp(a.value), math.exp(a.value), a)

def get_gradient_helper(currentNode: MyScalar, acc_grad) -> dict:
    if currentNode.parent is None:
        return {currentNode: acc_grad}

    acc_dict = get_gradient_helper(currentNode.parent, acc_grad * currentNode.grad)
    acc_dict[currentNode] = acc_grad

    return acc_dict

def get_gradient(a) -> dict:
    return get_gradient_helper(a, 1.0)

def test_get_gradient():
    
    a = MyScalar(3.0, 1.0, None)
    a_tensor = torch.tensor(a.value, requires_grad = True)

    b = pow(a, 2)
    b_tensor = a_tensor ** 2
    b_tensor.retain_grad()

    c = pow(b, 4)
    c_tensor = b_tensor ** 4
    c_tensor.retain_grad()

    c_tensor.backward()
    gradients = get_gradient(c)

    torch.testing.assert_close(gradients[a], a_tensor.grad.item())
    torch.testing.assert_close(gradients[b], b_tensor.grad.item())
    torch.testing.assert_close(gradients[c], c_tensor.grad.item())
    pass


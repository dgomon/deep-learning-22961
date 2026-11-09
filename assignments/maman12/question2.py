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
    return MyScalar(math.sin(a.value), math.cos(a.value), a)


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


def test_sequence1():
    a = MyScalar(2.0, 1.0, None)
    a_tensor = torch.tensor(a.value, requires_grad=True)

    # Testing sequence:
    # exp, sin, cos, pow, mul, plus

    b = exp(a)
    b_tensor = torch.exp(a_tensor)
    b_tensor.retain_grad()

    c = sin(b)
    c_tensor = torch.sin(b_tensor)
    c_tensor.retain_grad()

    d = pow(c, 1.5)
    d_tensor = torch.pow(c_tensor, 1.5)
    d_tensor.retain_grad()

    e = mul(d, 3.14)
    e_tensor = d_tensor * 3.14
    e_tensor.retain_grad()

    f = plus(e, 5.12)
    f_tensor = e_tensor + 5.12
    f_tensor.retain_grad()

    grads = get_gradient(f)
    f_tensor.backward()

    assert math.isclose(grads[a], a_tensor.grad.item(), rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(grads[b], b_tensor.grad.item(), rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(grads[c], c_tensor.grad.item(), rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(grads[d], d_tensor.grad.item(), rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(grads[e], e_tensor.grad.item(), rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(grads[f], f_tensor.grad.item(), rel_tol=1e-4, abs_tol=1e-4)
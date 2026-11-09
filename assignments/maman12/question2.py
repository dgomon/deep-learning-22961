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
    return MyScalar(math.cos(a.value), -1 * math.sin(a.value), a)


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
    # exp, sin, pow, mul, plus, cos

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

    g = cos(f)
    g_tensor = torch.cos(f_tensor)
    g_tensor.retain_grad()

    grads = get_gradient(g)
    g_tensor.backward()

    for (scalar, tensor) in zip((a, b, c, d, e, f, g),
                              (a_tensor, b_tensor, c_tensor, d_tensor, e_tensor, f_tensor, g_tensor)):
        assert math.isclose(grads[scalar], tensor.grad.item(), rel_tol=1e-4, abs_tol=1e-4)

def test_sequence2():
        a = MyScalar(1.5, 1.0, None)
        a_tensor = torch.tensor(a.value, requires_grad=True)

        # Testing sequence:
        # sin, exp, pow, plus, cos, mul

        b = sin(a)
        b_tensor = torch.sin(a_tensor)
        b_tensor.retain_grad()

        c = exp(b)
        c_tensor = torch.exp(b_tensor)
        c_tensor.retain_grad()

        d = pow(c, 1.5)
        d_tensor = torch.pow(c_tensor, 1.5)
        d_tensor.retain_grad()

        e = plus(d, 5.12)
        e_tensor = d_tensor + 5.12
        e_tensor.retain_grad()

        f = sin(e)
        f_tensor = torch.sin(e_tensor)
        f_tensor.retain_grad()

        g = mul(f, 3.14)
        g_tensor = f_tensor * 3.14
        g_tensor.retain_grad()

        grads = get_gradient(g)
        g_tensor.backward()

        for (scalar, tensor) in zip((a, b, c, d, e, f, g),
                                  (a_tensor, b_tensor, c_tensor, d_tensor, e_tensor, f_tensor, g_tensor)):
            assert math.isclose(grads[scalar], tensor.grad.item(), rel_tol=1e-4, abs_tol=1e-4)

from __future__ import annotations

import math

import torch


class MyScalar:
    def __init__(self, value, grad, parent):
        self.value = value
        self.grad = grad
        self.parent = parent

def plus(a, n):
    return MyScalar(a.value + n, 1, a)

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
    return get_gradient_helper(a, 1)

def test_get_gradient():
    
    a = MyScalar(3.0, 1, None)
    a_tensor = torch.tensor(a.value, requires_grad = True)

    b = pow(a, 2)
    b_tensor = a_tensor ** 2
    b_tensor.retain_grad()

    # gradient_b = get_gradient(b)
    # b_tensor.backward()
    # pass

    c = pow(b, 4)
    c_tensor = b_tensor ** 4
    c_tensor.retain_grad()

    gradient_c = get_gradient(c)
    c_tensor.backward()
    # print(b_tensor.grad) # todo: why returns tensor(5.)?

    pass


    d = pow(c, 3)
    d_tensor = c_tensor ** 3

    d_tensor.backward()

    torch.testing.assert_close(d.value, d_tensor.item())
    torch.testing.assert_close(d.grad, d_tensor.grad)

    torch.testing.assert_close(c.value, c_tensor.item())
    torch.testing.assert_close(c.grad, c_tensor.grad)


# def main():
#     a = MyScalar(2, 1, None)
#     b = power(a,2) #a^2
#     c = exp(b)
#     d = get_gradient(c)
#     print(d)
#     pass

# if __name__ == "__main__":
#   main()

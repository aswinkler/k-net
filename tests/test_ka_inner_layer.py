# test_ka_inner_layer.py

import numpy as np
from knetdual.ka_inner_layer import KAInnerLayer
from knetdual.dual_tensor import DualTensor

def test_ka_inner_real():
    np.random.seed(0)
    input_dim = 2
    output_dim = 5
    knots = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]

    layer = KAInnerLayer(input_dim, output_dim, knots)
    x = np.array([[0.5, 1.5], [1.0, 2.0]])

    y = layer.forward(x)
    assert y.shape == (2, output_dim)
    print("Real input test passed.")

def test_ka_inner_dual():
    np.random.seed(0)
    input_dim = 2
    output_dim = 5
    knots = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]

    layer = KAInnerLayer(input_dim, output_dim, knots)

    real = np.array([[0.5, 1.5], [1.0, 2.0]])
    dual = np.ones_like(real)
    x_dual = DualTensor(real, dual)

    y = layer.forward(x_dual)
    assert isinstance(y, DualTensor)
    assert y.real.shape == (2, output_dim)
    assert y.dual.shape == (2, output_dim)
    print("Dual input test passed.")

if __name__ == "__main__":
    test_ka_inner_real()
    test_ka_inner_dual()
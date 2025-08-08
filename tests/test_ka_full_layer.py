# test_ka_full_layer.py

import numpy as np
from ka_full_layer import KAFullLayer
from dual_tensor import DualTensor

def test_ka_full_real():
    knots = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
    model = KAFullLayer(input_dim=2, output_dim=3, knots=knots)

    x = np.random.rand(4, 2)  # batch_size = 4
    y = model.forward(x)
    assert isinstance(y, DualTensor)
    assert y.real.shape == (4, 3)
    print("KAFullLayer real input test passed.")

def test_ka_full_dual():
    knots = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
    model = KAFullLayer(input_dim=2, output_dim=3, knots=knots)

    real = np.random.rand(4, 2)
    dual = np.ones_like(real)
    x = DualTensor(real, dual)
    y = model.forward(x)
    assert y.real.shape == (4, 3)
    assert y.dual.shape == (4, 3)
    print("KAFullLayer dual input test passed.")

if __name__ == "__main__":
    test_ka_full_real()
    test_ka_full_dual()
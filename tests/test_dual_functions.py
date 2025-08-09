# test_dual_functions.py

import numpy as np
from knetdual.dual_tensor import DualTensor
from knetdual.dual_functions import sin, cos, exp, log, relu

def test_exp():
    x = DualTensor([0.0, 1.0], [1.0, 2.0])
    y = exp(x)
    assert np.allclose(y.real, np.exp([0.0, 1.0]))
    assert np.allclose(y.dual, np.exp([0.0, 1.0]) * [1.0, 2.0])

def test_log():
    x = DualTensor([1.0, np.e], [1.0, 2.0])
    y = log(x)
    expected_real = np.log([1.0, np.e])
    expected_dual = [1.0 / 1.0 * 1.0, 1.0 / np.e * 2.0]
    assert np.allclose(y.real, expected_real)
    assert np.allclose(y.dual, expected_dual)

def test_sin():
    x = DualTensor([0.0, np.pi/2], [1.0, 1.0])
    y = sin(x)
    assert np.allclose(y.real, np.sin([0.0, np.pi/2]))
    assert np.allclose(y.dual, np.cos([0.0, np.pi/2]))

def test_relu():
    x = DualTensor([-1.0, 2.0], [5.0, 3.0])
    y = relu(x)
    assert np.allclose(y.real, [0.0, 2.0])
    assert np.allclose(y.dual, [0.0, 3.0])

if __name__ == "__main__":
    test_exp()
    test_log()
    test_sin()
    test_relu()
    print("All DualFunction tests passed.")

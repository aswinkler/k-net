# test_dual_tensor.py
from knetdual.dual_tensor import DualTensor
import numpy as np

def test_addition():
    a = DualTensor([1, 2], [0.1, 0.2])
    b = DualTensor([3, 4], [0.3, 0.4])
    result = a + b
    assert np.allclose(result.real, [4, 6])
    assert np.allclose(result.dual, [0.4, 0.6])

def test_scalar_add():
    a = DualTensor([1, 2], [0.1, 0.2])
    result = a + 5
    assert np.allclose(result.real, [6, 7])
    assert np.allclose(result.dual, [0.1, 0.2])

def test_multiplication():
    a = DualTensor([1, 2], [0.1, 0.2])
    b = DualTensor([3, 4], [0.3, 0.4])
    result = a * b
    assert np.allclose(result.real, [3, 8])
    assert np.allclose(result.dual, [0.6, 1.6])

def test_division():
    a = DualTensor([2, 4], [1, 2])
    b = DualTensor([2, 2], [0.5, 1])
    result = a / b
    expected_real = [1, 2]
    expected_dual = [(1*2 - 2*0.5)/4, (2*2 - 4*1)/4]
    assert np.allclose(result.real, expected_real)
    assert np.allclose(result.dual, expected_dual)

if __name__ == "__main__":
    test_addition()
    test_scalar_add()
    test_multiplication()
    test_division()
    print("All DualTensor tests passed.")

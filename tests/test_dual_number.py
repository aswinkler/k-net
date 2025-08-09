# test_dual_number.py
from knetdual.dual_number import DualNumber

def test_addition():
    a = DualNumber(2, 3)
    b = DualNumber(1, 4)
    assert (a + b).real == 3 and (a + b).dual == 7

def test_multiplication():
    a = DualNumber(2, 3)
    b = DualNumber(1, 4)
    result = a * b  # (2)(1) + (2)(4) + (3)(1)ε = 2 + 11ε
    assert abs(result.real - 2) < 1e-9 and abs(result.dual - 11) < 1e-9

def test_scalar_operations():
    a = DualNumber(2, 3)
    b = 5
    assert (a + b).real == 7
    assert (b * a).dual == 15

def test_division():
    a = DualNumber(2, 3)
    b = DualNumber(4, 1)
    result = a / b
    expected_real = 0.5
    expected_dual = (3 * 4 - 2 * 1) / (4**2)  # (12 - 2)/16 = 0.625
    assert abs(result.real - expected_real) < 1e-9
    assert abs(result.dual - expected_dual) < 1e-9

if __name__ == "__main__":
    test_addition()
    test_multiplication()
    test_scalar_operations()
    test_division()
    print("All DualNumber tests passed.")
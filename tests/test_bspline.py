# test_bspline.py

from knetdual.bspline import BSplineBasis
from knetdual.dual_number import DualNumber
import numpy as np

def test_scalar_eval():
    knots = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]  # grado 3 => n = len(knots) - 4 = 6
    spline = BSplineBasis(knots, degree=3)
    values = spline.all_basis(1.5)
    assert np.isclose(sum(values), 1.0), "Partición de la unidad no cumplida."

def test_dual_eval():
    knots = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
    spline = BSplineBasis(knots, degree=3)
    x = DualNumber(1.5, 1.0)  # valor + derivada
    values = spline.all_basis(x)
    assert isinstance(values[0], DualNumber)
    assert np.isclose(sum([v.real for v in values]), 1.0)

if __name__ == "__main__":
    test_scalar_eval()
    test_dual_eval()
    print("B-spline tests passed.")

# bspline.py

import numpy as np
from knetdual.dual_number import DualNumber

class BSplineBasis:
    """
    Representa una base B-spline de grado k con una secuencia de nudos.

    Atributos:
        knots (np.ndarray): Vector de nudos (ordenado no decreciente)
        degree (int): Grado del spline (por defecto cúbico: 3)
    """

    def __init__(self, knots, degree=3):
        self.knots = np.array(knots, dtype=float)
        self.degree = degree
        self.n = len(self.knots) - degree - 1  # número de funciones base

        if self.n <= 0:
            raise ValueError("El número de funciones base debe ser mayor que cero.")

    def evaluate(self, i, k, x):
        """
        Evalúa la i-ésima función B-spline de grado k en el punto x usando recursión.

        Args:
            i (int): índice de la función base
            k (int): grado actual
            x (float o DualNumber): punto de evaluación

        Returns:
            float o DualNumber: valor de la base en x
        """

        # print("Type x in B-spline: ", type(x))
        t = self.knots

        if k == 0:
            if isinstance(x, DualNumber):
                return DualNumber(1.0 if t[i] <= x.real < t[i+1] else 0.0, 0.0)
            else:
                return 1.0 if t[i] <= x < t[i+1] else 0.0

        denom1 = t[i + k] - t[i]
        denom2 = t[i + k + 1] - t[i + 1]

        term1 = 0.0
        if denom1 > 0:
            term1 = (x - t[i]) / denom1 * self.evaluate(i, k - 1, x)

        term2 = 0.0
        if denom2 > 0:
            term2 = (t[i + k + 1] - x) / denom2 * self.evaluate(i + 1, k - 1, x)

        return term1 + term2

    def all_basis(self, x):
        """
        Evalúa todas las funciones base en un punto x.

        Args:
            x (float o DualNumber)

        Returns:
            list of float o DualNumber: valores de cada base en x
        """
        return [self.evaluate(i, self.degree, x) for i in range(self.n)]

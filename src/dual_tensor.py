# dual_tensor.py

import numpy as np

class DualTensor:
    """
    Representa un tensor cuyos elementos son números duales a + bε,
    separados en dos arreglos NumPy: real y dual.

    Atributos:
        real (np.ndarray): Parte real del tensor.
        dual (np.ndarray): Parte dual del tensor (derivada).
    """

    def __init__(self, real, dual=None):
        """
        Inicializa un DualTensor con parte real y parte dual.
        Si la parte dual no se especifica, se asume cero.

        Args:
            real (array-like): Parte real.
            dual (array-like or None): Parte dual. Si es None, se llena con ceros.
        """
        self.real = np.array(real, dtype=float)
        if dual is None:
            self.dual = np.zeros_like(self.real)
        else:
            self.dual = np.array(dual, dtype=float)
        
        if self.real.shape != self.dual.shape:
            raise ValueError("Las partes real y dual deben tener la misma forma.")

    def __add__(self, other):
        if isinstance(other, DualTensor):
            return DualTensor(self.real + other.real, self.dual + other.dual)
        else:
            return DualTensor(self.real + other, self.dual)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, DualTensor):
            real_part = self.real * other.real
            dual_part = self.real * other.dual + self.dual * other.real
            return DualTensor(real_part, dual_part)
        else:
            return DualTensor(self.real * other, self.dual * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return DualTensor(-self.real, -self.dual)

    def __truediv__(self, other):
        if isinstance(other, DualTensor):
            real_part = self.real / other.real
            dual_part = (self.dual * other.real - self.real * other.dual) / (other.real ** 2)
            return DualTensor(real_part, dual_part)
        else:
            return DualTensor(self.real / other, self.dual / other)

    def __repr__(self):
        return f"DualTensor(real={self.real}, dual={self.dual})"

    def shape(self):
        return self.real.shape

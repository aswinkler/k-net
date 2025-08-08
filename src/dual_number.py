# dual_number.py

class DualNumber:
    def __init__(self, real, dual=0.0):
        self.real = float(real)
        self.dual = float(dual)

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        else:
            return DualNumber(self.real + other, self.dual)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(
                self.real * other.real,
                self.real * other.dual + self.dual * other.real
            )
        else:
            return DualNumber(self.real * other, self.dual * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return DualNumber(-self.real, -self.dual)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(
                self.real / other.real,
                (self.dual * other.real - self.real * other.dual) / (other.real ** 2)
            )
        else:
            return DualNumber(self.real / other, self.dual / other)

    def __repr__(self):
        return f"{self.real} + {self.dual}ε"

    def __str__(self):
        return f"{self.real} + {self.dual}ε"
        
    def format(self, digits=4):
        return f"{round(self.real, digits)} + {round(self.dual, digits)}ε"

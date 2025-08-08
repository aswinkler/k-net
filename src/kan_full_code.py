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
# dual_functions.py

import numpy as np
from dual_tensor import DualTensor

def sin(x: DualTensor) -> DualTensor:
    """
    Aplica la función seno a un DualTensor.

    Derivada: d/dx(sin(x)) = cos(x)
    """
    real_part = np.sin(x.real)
    dual_part = np.cos(x.real) * x.dual
    return DualTensor(real_part, dual_part)

def cos(x: DualTensor) -> DualTensor:
    """
    Aplica la función coseno a un DualTensor.

    Derivada: d/dx(cos(x)) = -sin(x)
    """
    real_part = np.cos(x.real)
    dual_part = -np.sin(x.real) * x.dual
    return DualTensor(real_part, dual_part)

def exp(x: DualTensor) -> DualTensor:
    """
    Aplica la exponencial a un DualTensor.

    Derivada: d/dx(exp(x)) = exp(x)
    """
    real_part = np.exp(x.real)
    dual_part = real_part * x.dual
    return DualTensor(real_part, dual_part)

def log(x: DualTensor) -> DualTensor:
    """
    Aplica el logaritmo natural a un DualTensor.

    Derivada: d/dx(log(x)) = 1/x
    """
    real_part = np.log(x.real)
    dual_part = (1 / x.real) * x.dual
    return DualTensor(real_part, dual_part)

def relu(x: DualTensor) -> DualTensor:
    """
    Aplica la función ReLU (Rectified Linear Unit).

    Derivada: 1 si x > 0, 0 si x <= 0
    """
    real_part = np.maximum(0, x.real)
    mask = (x.real > 0).astype(float)
    dual_part = mask * x.dual
    return DualTensor(real_part, dual_part)
# bspline.py

import numpy as np
from dual_number import DualNumber

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
# ka_inner_layer.py

import numpy as np
from dual_tensor import DualTensor
from dual_number import DualNumber
from bspline import BSplineBasis

class KAInnerLayer:
    """
    Capa interna del teorema de Kolmogorov-Arnold.
    Cada función φ_q(x) es una suma ponderada de funciones B-spline univariadas aplicadas a cada entrada x_p.
    """

    def __init__(self, input_dim, output_dim, knots):
        """
        Inicializa la capa KA interna.

        Args:
            input_dim (int): Número de variables de entrada (n)
            output_dim (int): Número de funciones φ_q (por convención: 2n+1)
            knots (list or np.ndarray): Secuencia de nudos para los B-splines
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = 3

        # Creamos una matriz de bases [q][p] -> BSplineBasis
        self.bases = [[BSplineBasis(knots, self.degree) for p in range(input_dim)]
                      for q in range(output_dim)]

        # Inicializamos pesos aleatorios λ_{q,p,i}
        n_bases = len(knots) - self.degree - 1
        self.weights = np.random.randn(output_dim, input_dim, n_bases)

    def forward(self, x):
        """
        Evalúa la capa KA interna para un batch de entradas.
    
        Args:
            x (np.ndarray or DualTensor): Entrada de forma (batch_size, input_dim)
    
        Returns:
            DualTensor: salida de forma (batch_size, output_dim)
        """
       
        if isinstance(x, DualTensor):
            real_x = x.real
            dual_x = x.dual
            use_dual = True
        else:
            real_x = x
            dual_x = None
            use_dual = False
    
        batch_size = real_x.shape[0]
        output_real = np.zeros((batch_size, self.output_dim))
        output_dual = np.zeros((batch_size, self.output_dim)) if use_dual else None
    
        for b in range(batch_size):
            for q in range(self.output_dim):
                total = 0.0
                dual_total = 0.0 if use_dual else None
                for p in range(self.input_dim):
                    xp = real_x[b, p]
                    x_dual = dual_x[b, p] if use_dual else 0.0
                    x_input = DualNumber(xp, x_dual) if use_dual else xp
    
                    basis_values = self.bases[q][p].all_basis(x_input)
                    weights = self.weights[q, p, :]
    
                    # Composición con soporte para DualNumber en los pesos
                    if isinstance(weights[0], DualNumber):
                        partial = DualNumber(0.0, 0.0)
                        for w, bval in zip(weights, basis_values):
                            if isinstance(bval, DualNumber):
                                partial += w * bval
                            else:
                                partial += w * DualNumber(bval, 0.0)
                        total += partial.real
                        dual_total += partial.dual
                    else:
                        if use_dual:
                            real_part = sum(w * b.real for w, b in zip(weights, basis_values))
                            dual_part = sum(w * b.dual for w, b in zip(weights, basis_values))
                            total += real_part
                            dual_total += dual_part
                        else:
                            total += sum(w * b for w, b in zip(weights, basis_values))
    
                output_real[b, q] = total
                if use_dual:
                    output_dual[b, q] = dual_total
    
        if use_dual:
            return DualTensor(output_real, output_dual)
        else:
            return output_real
# ka_outer_layer.py

import numpy as np
from dual_number import DualNumber
from dual_tensor import DualTensor

class KAOuterLayer:
    """
    Capa externa φ_out del modelo KA: combina funciones φ_q(x) usando pesos y una activación opcional.
    """

    def __init__(self, input_dim, activation=None):
        self.input_dim = input_dim
        self.weights = np.random.randn(input_dim)
        self.activation = activation if activation else (lambda x: x)  # Identidad

    def forward(self, x: DualTensor) -> DualTensor:
        """
        Args:
            x (DualTensor): entrada de forma (batch_size, input_dim)

        Returns:
            DualTensor de salida de forma (batch_size,)
        """
        #print("Ejecutando KAOuterLayer.forward")
        #print("Type before activation: ", type(x))
        # asegurar que x es DualTensor
        if not isinstance(x, DualTensor):
            x = DualTensor(x)
            
        if x.real.shape[1] != self.input_dim:
            raise ValueError("Dimensión incorrecta de entrada")

        activated = self.activation(x)
        
        #print("Type after activation: ", type(activated))
        # asegurar que la activación devuelve DualTensor
        if not isinstance(activated, DualTensor):
            activated = DualTensor(activated)
            
        # Detectar si estamos usando DualNumbers como pesos
        # print("tipo de self.weights[0] en KAOuterLayer.forward", type(self.weights[0]))             
        if isinstance(self.weights[0], DualNumber):
            # Evaluar producto real y dual a mano
            # print("Evaluando producto real y dual manualmente")
            batch_size = x.real.shape[0]
            out_real = np.zeros(batch_size)
            out_dual = np.zeros(batch_size)
            for b in range(batch_size):
                for i in range(self.input_dim):
                    w = self.weights[i]
                    val = DualNumber(activated.real[b, i], activated.dual[b, i])
                    result = w * val
                    out_real[b] += result.real
                    out_dual[b] += result.dual
            return DualTensor(out_real, out_dual)
        else:
            real = np.sum(activated.real * self.weights, axis=1)
            dual = np.sum(activated.dual * self.weights, axis=1)
            return DualTensor(real, dual)

    def set_weights(self, new_weights):
        new_weights = np.array(new_weights, dtype=float)
        if new_weights.shape[0] != self.input_dim:
            raise ValueError("Tamaño incompatible de pesos")
        self.weights = new_weights




def forward(self, x):
    """
    Args:
        x (DualTensor or np.ndarray): Entrada (batch_size, input_dim)

    Returns:
        DualTensor: salida (batch_size,)
    """
    if not isinstance(x, DualTensor):
        x = DualTensor(x)

    if x.real.shape[1] != self.input_dim:
        raise ValueError("Dimensión incorrecta de entrada")

    activated = self.activation(x)

    # Detectar si estamos usando DualNumbers como pesos
    if isinstance(self.weights[0], DualNumber):
        # Evaluar producto real y dual a mano
        batch_size = x.real.shape[0]
        out_real = np.zeros(batch_size)
        out_dual = np.zeros(batch_size)
        for b in range(batch_size):
            for i in range(self.input_dim):
                w = self.weights[i]
                val = DualNumber(activated.real[b, i], activated.dual[b, i])
                result = w * val
                out_real[b] += result.real
                out_dual[b] += result.dual
        return DualTensor(out_real, out_dual)
    else:
        # Caso normal: pesos reales
        real = np.sum(activated.real * self.weights, axis=1)
        dual = np.sum(activated.dual * self.weights, axis=1)
        return DualTensor(real, dual)
# ka_full_layer.py

import numpy as np
from ka_inner_layer import KAInnerLayer
from ka_outer_layer import KAOuterLayer
from dual_tensor import DualTensor

class KAFullLayer:
    """
    Extensión multisalida del modelo KA: combina múltiples bloques KAInner → KAOuter.

    Para cada salida, se define una red independiente que sigue la representación del teorema de Kolmogorov-Arnold.
    """

    def __init__(self, input_dim, output_dim, knots, activation=None):
        """
        Args:
            input_dim (int): Número de variables de entrada (n)
            output_dim (int): Número de salidas requeridas (m)
            knots (list): Nudos comunes para los B-splines
            activation (func): Activación externa Φ_q (por ejemplo ReLU)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 2 * input_dim + 1  # como indica el teorema

        # Crear un par (KAInner, KAOuter) por cada salida
        self.blocks = [
            (
                KAInnerLayer(input_dim, self.hidden_dim, knots),
                KAOuterLayer(self.hidden_dim, activation)
            )
            for _ in range(output_dim)
        ]

    def forward(self, x):
        """
        Ejecuta el modelo para un lote de entradas.

        Args:
            x (np.ndarray o DualTensor): Entrada de forma (batch_size, input_dim)

        Returns:
            DualTensor: salida de forma (batch_size, output_dim)
        """
        #print("Ejecutando KAFullLayer.forward")
        outputs_real = []
        outputs_dual = []

        for inner, outer in self.blocks:
            z = inner.forward(x)
            y = outer.forward(z)
            outputs_real.append(y.real)
            outputs_dual.append(y.dual)

        real = np.stack(outputs_real, axis=1)  # (batch_size, output_dim)
        dual = np.stack(outputs_dual, axis=1)
        return DualTensor(real, dual)# kan_model.py

from ka_full_layer import KAFullLayer
from dual_tensor import DualTensor

class KANetwork:
    """
    Red neuronal basada completamente en bloques KAFullLayer.
    Arquitectura definida por capas (entradas y salidas).
    """

    def __init__(self, layer_dims, knots, activation=None):
        """
        Args:
            layer_dims (list): lista de dimensiones de capa, ej. [13, 5, 3]
            knots (list): nudos B-spline
            activation (func): función de activación externa (ReLU, sigmoide, etc.)
        """
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(KAFullLayer(
                input_dim=layer_dims[i],
                output_dim=layer_dims[i + 1],
                knots=knots,
                activation=activation
            ))
            
    def forward(self, x):
        """
        Propagación directa por la red.

        Args:
            x (np.ndarray o DualTensor): Entrada (batch_size, 4)

        Returns:
            DualTensor: Salida (batch_size, 3)
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x):
        """ Propagación directa para salida real sin derivadas """
        out = self.forward(DualTensor(x))
        return out.real
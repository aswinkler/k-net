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

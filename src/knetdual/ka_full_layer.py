# ka_full_layer.py

import numpy as np
from knetdual.ka_inner_layer import KAInnerLayer
from knetdual.ka_outer_layer import KAOuterLayer
from knetdual.dual_tensor import DualTensor

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
        return DualTensor(real, dual)
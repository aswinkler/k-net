# ka_outer_layer.py

import numpy as np
from knetdual.dual_number import DualNumber
from knetdual.dual_tensor import DualTensor

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

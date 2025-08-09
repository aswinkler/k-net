# kan_model.py

from knetdual.ka_full_layer import KAFullLayer
from knetdual.dual_tensor import DualTensor

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
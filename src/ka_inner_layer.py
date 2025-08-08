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

if __name__ == "__main__":
    import numpy as np

    from dual_tensor import DualTensor
    from kan_model import KANetwork
    x = np.random.rand(2, 4)
    model = KANetwork(knots=[0, 0, 0, 0, 1, 2, 3, 3, 3, 3])
    y = model.forward(x)

    print("Output real:", y.real)
    print("Output dual:", y.dual)
# train_ka_network.py

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

from dual_tensor import DualTensor
from ka_inner_layer import KAInnerLayer
from ka_outer_layer import KAOuterLayer
from dual_functions import relu
from dual_number import DualNumber

def softmax(x: DualTensor):
    exps = np.exp(x.real - np.max(x.real, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    return DualTensor(probs, x.dual)

def cross_entropy(pred: DualTensor, y: np.ndarray):
    eps = 1e-9
    logp = np.log(pred.real + eps)
    loss = -np.sum(y * logp, axis=1)  # por muestra
    grad = -y / (pred.real + eps)
    dloss = np.sum(grad * pred.dual, axis=1)
    return DualTensor(loss, dloss)

def one_hot(y, num_classes=3):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

def train_step(x_batch, y_batch, model_inner, model_outer, lr=0.01):
    batch_size = x_batch.shape[0]
    input_dim = x_batch.shape[1]

    # Paso directo sin perturbar pesos
    x_dual = DualTensor(x_batch, np.zeros_like(x_batch))
    z = model_inner.forward(x_dual)
    y_hat = softmax(model_outer.forward(z))
    loss = cross_entropy(y_hat, y_batch)

    # Gradientes por diferenciación hacia adelante
    grad_outer = np.zeros_like(model_outer.weights)

    for i in range(len(model_outer.weights)):
        w = model_outer.weights.copy()
        w[i] += 1e-12  # ε
        perturbed = DualTensor(z.real, z.dual.copy())
        model_outer.weights = w
        y_hat_eps = softmax(model_outer.forward(perturbed))
        loss_eps = cross_entropy(y_hat_eps, y_batch)
        grad_outer[i] = np.mean(loss_eps.dual)

    model_outer.weights -= lr * grad_outer
    return np.mean(loss.real)

def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    X = StandardScaler().fit_transform(X)
    Y = one_hot(y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    input_dim = X.shape[1]
    hidden_dim = 2 * input_dim + 1

    knots = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]

    inner = KAInnerLayer(input_dim=input_dim, output_dim=hidden_dim, knots=knots)
    outer = KAOuterLayer(input_dim=hidden_dim)

    for epoch in range(100):
        loss = train_step(X_train, Y_train, inner, outer, lr=0.05)
        print(f"Epoch {epoch:03d}: Loss = {loss:.4f}")

if __name__ == "__main__":
    main()
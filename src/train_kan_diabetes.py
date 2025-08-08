# train_kan_iris.py

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dual_tensor import DualTensor
from kan_model import KANetwork
from dual_functions import relu
from dual_number import DualNumber


def softmax(x: DualTensor):
    # parte real
    exps = np.exp(x.real - np.max(x.real, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    # parte dual: ds_j = sum_i d_softmax_j/dx_i * dx_i
    dot = np.zeros_like(x.dual)

    for b in range(x.real.shape[0]):  # por cada muestra
        s = probs[b]                  # vector softmax
        dx = x.dual[b]                # derivadas duales de entrada
        J = np.diag(s) - np.outer(s, s)  # jacobiano softmax
        dot[b] = J @ dx               # producto jacobiano * derivadas
    # print("Softmax output (real): ",probs[:5])
    # print("Softmax output (dual): ",dot[:5])         
    return DualTensor(probs, dot)  # Suponemos misma derivada para simplificar


def cross_entropy(pred: DualTensor, y_true: np.ndarray):
    eps = 1e-9
    logp = np.log(pred.real + eps)
    loss = -np.sum(y_true * logp, axis=1)
    grad = -y_true / (pred.real + eps)
    dloss = np.sum(grad * pred.dual, axis=1)
    return DualTensor(loss, dloss)

def mse_dual(pred: DualTensor, y_true: np.ndarray):
    error = pred.real - y_true
    loss = np.mean(error ** 2, axis=1)
    dloss = 2 * np.mean(error * pred.dual, axis=1)
    return DualTensor(loss, dloss)


def one_hot(y, num_classes=3):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh
    

def train_step(X_batch, Y_batch, model: KANetwork, lr=0.01):
    batch_size = X_batch.shape[0]
    x_dual = DualTensor(X_batch, np.zeros_like(X_batch))
    
    # === ENTRENAMIENTO DE KAInnerLayer ===
    for layer in model.layers:
        Q = layer.output_dim
        P = layer.input_dim
        K = layer.blocks[0][0].weights.shape[2]
        
        print("Entrenando capa interna")
        
        #entrenamiento de la capa interna del bloque
        for q in range(Q):
            block_inner = layer.blocks[q][0]
            H = block_inner.weights.shape[0]
            for h in range(H):
                for p in range(P):
                    for i in range(K):
                        #print("Capa interna: q = ", q, " h = ", h, " p = ", p, " i = ", i) 
                        #block_inner = layer.blocks[q][0]

                        # Copia original de los pesos
                        original_weights = block_inner.weights.copy()

                        # Versión modificable con DualNumber
                        modified_weights = original_weights.astype(object)

                        for idx in range(K):
                            val = original_weights[q, p, idx]
                            modified_weights[q, p, idx] = DualNumber(val, 1.0 if idx == i else 0.0)

                    block_inner.weights = modified_weights

                    preds = model.forward(x_dual)
                    loss = mse_dual(preds, Y_batch)
                    grad = np.mean(loss.dual)

                    # Restaurar y actualizar
                    original_weights[q, p, i] -= lr * grad
                    block_inner.weights = original_weights

        # === ENTRENAMIENTO DE KAOuterLayer ===
        # Q = layer.output_dim
        N = layer.blocks[0][1].weights.shape[0]  # número de entradas internas (2n+1)
        
        print("Entrenando capas externas")

        for q in range(Q):
            block_outer = layer.blocks[q][1]
            original_weights = block_outer.weights.copy()

            for i in range(N):
                #print("Capa externa: ", layer, " q = ", q, " i = ", i) 
                dual_weights = [DualNumber(w, 0.0) for w in original_weights]
                dual_weights[i] = DualNumber(original_weights[i], 1.0)

                block_outer.weights = dual_weights
                preds = model.forward(x_dual)
                loss = mse_dual(preds, Y_batch)
                grad = np.mean(loss.dual)

                original_weights[i] -= lr * grad

            block_outer.weights = original_weights

    return np.mean(loss.real)

def compute_accuracy(X, Y, model: KANetwork):
    """
    Evalúa el modelo en X y calcula la precisión comparando con etiquetas Y.
    """
    logits = model.forward(X)
    probs = softmax(logits).real
    predictions = np.argmax(probs, axis=1)
    targets = np.argmax(Y, axis=1)
    return np.mean(predictions == targets)

def compute_rmse(X, Y, model: KANetwork):
    """
    Evalúa el modelo en X y calcula la precisión comparando con etiquetas Y.
    """
    preds = model.forward(X).real
    return np.sqrt(np.mean(preds - Y) ** 2)

from sklearn.metrics import mean_squared_error as rmse_real_eval
import time

def train_with_logging(train_step_func, model, X_train, Y_train, X_test, Y_test, epochs=50, lr=0.05):
    """
    Ejecuta entrenamiento por múltiples épocas, registrando métricas.
    
    Args:
        train_step_func: función de entrenamiento (como train_step)
        model: instancia del modelo KANetwork
        X_train, Y_train: datos de entrenamiento
        X_test, Y_test: datos de prueba
        epochs: número de épocas
        lr: tasa de aprendizaje

    Returns:
        dict con 'loss', 'train_acc', 'test_acc'
    """
    loss_log = []
    train_rmse_log = []
    test_rmse_log = []
    time_log = []

    for epoch in range(epochs):
        start_time = time.time()
        loss = train_step_func(X_train, Y_train, model, lr=lr)
        loss_log.append(loss)

        # Forward pass con salida real

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        rmse_train = rmse_real_eval(Y_train, pred_train)
        rmse_test = rmse_real_eval(Y_test, pred_test)

        train_rmse_log.append(rmse_train)
        test_rmse_log.append(rmse_test)

        end_time = time.time()
        epoch_duration = end_time - start_time
        time_log.append(epoch_duration)

        print(f"Epoch {epoch:03d}: Loss = {loss:.4f} | Train RMSE = {rmse_train:.4f} | Test RMSE = {rmse_test:.4f} | Time = {epoch_duration: .2f} secs")

    
    save_metrics_to_csv(loss_log, train_rmse_log, test_rmse_log, time_log) 
    return {
        'loss': loss_log,
        'train_acc': train_rmse_log,
        'test_acc': test_rmse_log,
        'time' : time_log
    }

import matplotlib.pyplot as plt

def plot_training_metrics(metrics):
    """
    Grafica la pérdida y la precisión del entrenamiento y prueba.

    Args:
        metrics: diccionario con listas 'loss', 'train_acc', 'test_acc'
    """
    epochs = np.arange(len(metrics['loss']))

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Gráfico de pérdida
    ax[0].plot(epochs, metrics['loss'], label='Training Loss', color='red')
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training Loss over Epochs")
    ax[0].grid(True)
    ax[0].legend()

    # Gráfico de RMSE
    ax[1].plot(epochs, np.array(metrics['train_acc']) * 100, label='Train RMSE', color='blue')
    ax[1].plot(epochs, np.array(metrics['test_acc']) * 100, label='Test RMSE', color='green')
    ax[1].set_ylabel("RMSE")
    ax[1].set_xlabel("Epoch")
    ax[1].set_title("Training and Test RMSE over Epochs")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    
import csv
from datetime import datetime

def save_metrics_to_csv(losses, train_accuracies, test_accuracies, times,
    base_filename="training_metrics.csv"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{timestamp}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss", "Train RMSE", "Test RMSE", "Time"])
        for epoch, (loss, train_acc, test_acc, time) in enumerate(zip(losses, train_accuracies, test_accuracies, times), start=1):
            writer.writerow([epoch, loss, train_acc, test_acc, time])
    
def main():
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target.reshape(-1, 1)
    Y = y

    X = StandardScaler().fit_transform(X)
    Y = StandardScaler().fit_transform(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    knots = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
    print ("Entrenando con el dataset Diabetes, Red: [10, 2, 1]")
    model = KANetwork(layer_dims=[10, 2, 1], knots=knots, activation=relu)
    metrics = train_with_logging(train_step, model, X_train, Y_train, X_test, Y_test, epochs=50, lr=0.05)
    plot_training_metrics(metrics)

if __name__ == "__main__":
    main()



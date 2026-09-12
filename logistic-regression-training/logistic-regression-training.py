import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    size = len(X)
    weights = np.zeros(X.shape[1])
    biases = 0.0
    for i in range(steps):
        p = _sigmoid(np.matmul(X, weights) + biases)
        dldw = X.T @ (p - y) / len(X)
        dldb = np.mean(p - y)
        weights = weights - lr * dldw
        biases = biases - lr * dldb
    return weights, biases
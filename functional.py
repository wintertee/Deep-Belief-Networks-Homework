import cupy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -np.sum(y_true * np.log(y_pred + 1e-10), axis=1).mean()


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

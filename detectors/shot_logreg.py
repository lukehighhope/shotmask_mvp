import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _standardize_features(X, mean=None, std=None):
    X = np.asarray(X, dtype=np.float64)
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std, mean, std


def train_logreg(X, y, lr=0.1, epochs=2000, l2=1e-3, sample_weight=None):
    """
    Train binary logistic regression with L2 regularization.
    Returns model dict with weights/bias/mean/std.
    """
    Xs, mean, std = _standardize_features(X)
    y = np.asarray(y, dtype=np.float64)
    if sample_weight is None:
        sample_weight = np.ones_like(y, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    n_samples, n_features = Xs.shape
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    for _ in range(epochs):
        logits = Xs @ w + b
        p = _sigmoid(logits)
        weighted = (p - y) * sample_weight
        grad_w = (Xs.T @ weighted) / n_samples + l2 * w
        grad_b = np.mean(weighted)
        w -= lr * grad_w
        b -= lr * grad_b
    return {
        "type": "binary",
        "weights": w.tolist(),
        "bias": float(b),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def predict_logreg(model, X):
    Xs, _, _ = _standardize_features(X, mean=np.asarray(model["mean"]), std=np.asarray(model["std"]))
    w = np.asarray(model["weights"], dtype=np.float64)
    b = float(model["bias"])
    logits = Xs @ w + b
    return _sigmoid(logits)



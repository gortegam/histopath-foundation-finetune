import json, torch
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def save_label_map(classes, out_path):
    with open(out_path, 'w') as f:
        json.dump({'classes': classes}, f)

def load_label_map(path):
    with open(path, 'r') as f:
        return json.load(f)['classes']

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int) if y_prob.ndim == 1 else y_prob.argmax(1)
    metrics = {}
    if y_prob.ndim == 1:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics['auc'] = float('nan')
    metrics['acc'] = accuracy_score(y_true, y_pred)
    return metrics

class EarlyStopper:
    """Simple early stopping on a scalar score (higher is better).
    Usage:
        es = EarlyStopper(patience=3, min_delta=1e-4)
        if es.step(score):  # returns True when it's time to stop
            break
    """
    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0

    def step(self, value: float) -> bool:
        if self.best is None or (value - self.best) > self.min_delta:
            self.best = value
            self.count = 0
            return False
        else:
            self.count += 1
            return self.count >= self.patience


"""Метрики классификации подтипов (3 класса) и bootstrap-интервалы. Без torch."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

NAMES = ["ccRCC", "pRCC", "chRCC"]


def metrics(y, prob):
    pred = prob.argmax(1)
    out = {}
    try:
        out["macro_AUC"] = roc_auc_score(y, prob, multi_class="ovr", average="macro")
    except ValueError:
        out["macro_AUC"] = np.nan
    out["macro_F1"] = f1_score(y, pred, average="macro")
    out["accuracy"] = float((pred == y).mean())
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    for c, n in enumerate(NAMES):
        tp = cm[c, c]; fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp; tn = cm.sum() - tp - fn - fp
        out[f"sens_{n}"] = tp / (tp + fn) if tp + fn else np.nan
        out[f"spec_{n}"] = tn / (tn + fp) if tn + fp else np.nan
    return out


def bootstrap(y, prob, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 3:
            continue
        rows.append(metrics(y[idx], prob[idx]))
    return pd.DataFrame(rows)

"""Evaluation helpers: MAE over test set, sample predictions, optional plots."""
from __future__ import annotations
import numpy as np
import tensorflow as tf


def full_test_predictions(model, test_gen) -> tuple[np.ndarray, np.ndarray, float]:
    """Run predictions over the entire test generator and compute MAE manually."""
    y_true, y_pred = [], []
    test_gen.reset()
    for _ in range(len(test_gen)):
        Xb, yb = test_gen.next()
        preds = model.predict(Xb, verbose=0)
        y_true.append(yb)
        y_pred.append(preds.reshape(-1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return y_true, y_pred, mae
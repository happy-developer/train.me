import numpy as np
import pandas as pd

def maybe_apply_feature_scaler(X: pd.DataFrame, scaler):
    """Applique scaler.transform(X) si présent, sinon renvoie X."""
    if scaler is None:
        return X
    Xs = scaler.transform(X)
    return pd.DataFrame(Xs, columns=X.columns, index=X.index)

def maybe_inverse_target(y_pred_float: float, y_scaler):
    """
    Si un y_scaler est fourni, applique inverse_transform.
    Renvoie (y_pred_final: float, applied: bool).
    """
    if y_scaler is None:
        return y_pred_float, False
    try:
        y_final = float(y_scaler.inverse_transform(np.array([[y_pred_float]])).ravel()[0])
        return y_final, True
    except Exception:
        return y_pred_float, False

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def compute_metrics(y_true, y_pred):
    """
    Hesaplama metrikleri: R², RMSE, MAE
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
    
    Returns:
        dict: Metrik değerleri
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }

def compute_additional_metrics(y_true, y_pred):
    """
    Ek metrikler: MAPE, Explained Variance Score
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
    
    Returns:
        dict: Ek metrik değerleri
    """
    from sklearn.metrics import explained_variance_score
    
    # MAPE hesaplama (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    
    # Explained Variance Score
    ev_score = explained_variance_score(y_true, y_pred)
    
    return {
        'MAPE': mape,
        'Explained_Variance': ev_score
    } 
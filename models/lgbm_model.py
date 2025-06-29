import lightgbm as lgb
import numpy as np
import pandas as pd
from utils.metrics import compute_metrics
from utils.plots import plot_scatter, plot_shap, plot_residuals
from config import FEATURE_COLUMNS

class LGBMModel:
    def __init__(self, **kwargs):
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=-1,
            **kwargs
        )
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """LightGBM modelini eğitir"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        # TODO: Implement hyperparameter tuning for optimal performance.
    
    def predict(self, X_test):
        """Tahmin yapar"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        return self.model.predict(X_test)
    
    def evaluate(self, y_true, y_pred):
        """Model performansını değerlendirir"""
        return compute_metrics(y_true, y_pred)
    
    def plot_results(self, X_test, y_true, y_pred, output_dir):
        """Sonuçları görselleştirir ve PDF olarak kaydeder"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        # Scatter plot
        plot_scatter(y_true, y_pred, f"{output_dir}/lgbm_scatter.pdf", 
                    title="LightGBM: Predicted vs Actual")
        
        # SHAP plot (requires X_test as DataFrame with feature names)
        X_test_df = pd.DataFrame(X_test, columns=FEATURE_COLUMNS)
        plot_shap(self.model, X_test_df, f"{output_dir}/lgbm_shap.pdf")
        
        # Residuals plot
        plot_residuals(y_true, y_pred, f"{output_dir}/lgbm_residuals.pdf") 
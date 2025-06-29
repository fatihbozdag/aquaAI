from tpot import TPOTRegressor
import numpy as np
import pandas as pd
from utils.metrics import compute_metrics
from utils.plots import plot_scatter, plot_shap, plot_residuals
from config import FEATURE_COLUMNS

class TPOTModel:
    def __init__(self, **kwargs):
        self.model = TPOTRegressor(
            generations=5, # Reduced generations to mitigate overfitting
            population_size=10, # Reduced population size
            cv=3,
            random_state=42,
            verbosity=0,
            **kwargs
        )
        self.is_trained = False
        self.best_pipeline = None
    
    def train(self, X_train, y_train):
        """TPOT modelini eğitir"""
        self.model.fit(X_train, y_train)
        self.best_pipeline = self.model.fitted_pipeline_
        self.is_trained = True
        # TODO: For better generalization, consider increasing generations and population_size
        # with proper cross-validation and a larger dataset.
    
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
        plot_scatter(y_true, y_pred, f"{output_dir}/tpot_scatter.pdf", 
                    title="TPOT: Predicted vs Actual")
        
        # Try SHAP plot (if the best pipeline supports it)
        try:
            X_test_df = pd.DataFrame(X_test, columns=FEATURE_COLUMNS)
            plot_shap(self.best_pipeline, X_test_df, f"{output_dir}/tpot_shap.pdf")
        except Exception as e:
            print(f"TPOT SHAP plot error: {e}")
            # If SHAP fails, use feature importance
            from utils.plots import plot_feature_importance
            try:
                plot_feature_importance(self.best_pipeline, X_test_df, f"{output_dir}/tpot_importance.pdf")
            except Exception as fe:
                print(f"TPOT feature importance plot error: {fe}")
        
        # Residuals plot
        plot_residuals(y_true, y_pred, f"{output_dir}/tpot_residuals.pdf") 
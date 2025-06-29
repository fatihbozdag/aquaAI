from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from utils.metrics import compute_metrics
from utils.plots import plot_scatter, plot_residuals

class GPRModel:
    def __init__(self, **kwargs):
        # RBF kernel with optimized parameters
        kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0] * 4, (1e-2, 1e2))
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10, # Increased alpha for better generalization and to prevent overfitting
            random_state=42,
            **kwargs
        )
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """GPR modelini eğitir"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        # TODO: Implement hyperparameter tuning for kernel parameters and alpha for optimal performance.
    
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
        plot_scatter(y_true, y_pred, f"{output_dir}/gpr_scatter.pdf", 
                    title="GPR: Predicted vs Actual")
        
        # Residuals plot
        plot_residuals(y_true, y_pred, f"{output_dir}/gpr_residuals.pdf") 
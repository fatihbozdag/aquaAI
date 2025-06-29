#!/usr/bin/env python3
"""
Compare ANFIS performance before and after fixes with real water quality data simulation
"""
import numpy as np
import pandas as pd
from models.anfis_model import AnfisModel
from utils.metrics import compute_metrics
from sklearn.linear_model import LinearRegression

def simulate_water_quality_data(n_samples=150):
    """Simulate realistic water quality data based on domain knowledge"""
    np.random.seed(42)
    
    # Simulate realistic water quality parameters (already normalized [0,1])
    TP_norm = np.random.beta(2, 5, n_samples)  # Total Phosphorus (typically low)
    EC_norm = np.random.beta(3, 3, n_samples)  # Electrical Conductivity (moderate range)
    DO_norm = np.random.beta(5, 2, n_samples)  # Dissolved Oxygen (typically higher)
    tit_norm = np.random.beta(2, 3, n_samples)  # Target parameter base
    
    # Create realistic non-linear relationship for water quality index
    # Based on environmental science: DO positive, TP/EC negative correlation with quality
    quality_index = (
        0.4 * DO_norm +           # Dissolved oxygen improves quality
        -0.3 * TP_norm +          # Phosphorus reduces quality
        -0.2 * EC_norm +          # High conductivity reduces quality  
        0.1 * tit_norm +          # Base parameter
        0.1 * np.sin(np.pi * TP_norm * DO_norm) +  # Non-linear interaction
        0.05 * np.random.normal(0, 1, n_samples)   # Noise
    )
    
    # Normalize target to [0, 1] range
    quality_index = (quality_index - quality_index.min()) / (quality_index.max() - quality_index.min())
    
    X = np.column_stack([TP_norm, EC_norm, DO_norm, tit_norm])
    
    return X, quality_index

def test_original_vs_fixed():
    """Compare original (broken) vs fixed ANFIS"""
    print("ANFIS Performance Comparison: Original vs Fixed")
    print("=" * 60)
    
    # Generate realistic test data
    X_train, y_train = simulate_water_quality_data(200)
    X_test, y_test = simulate_water_quality_data(80)
    
    print(f"Training data: {X_train.shape}, Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"Test data: {X_test.shape}, Target range: [{y_test.min():.3f}, {y_test.max():.3f}]")
    print()
    
    # Test baseline linear regression
    print("Baseline Linear Regression:")
    print("-" * 30)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    lr_metrics = compute_metrics(y_test, y_pred_lr)
    
    for metric, value in lr_metrics.items():
        print(f"{metric}: {value:.4f}")
    print()
    
    # Test fixed ANFIS
    print("Fixed ANFIS Model:")
    print("-" * 30)
    anfis_fixed = AnfisModel(num_mfs_per_input=5)
    anfis_fixed.train(X_train, y_train)
    y_pred_anfis = anfis_fixed.predict(X_test)
    anfis_metrics = compute_metrics(y_test, y_pred_anfis)
    
    for metric, value in anfis_metrics.items():
        print(f"{metric}: {value:.4f}")
    print()
    
    # Simulate original broken behavior (with target inversion)
    print("Simulated Original (Broken) ANFIS:")
    print("-" * 35)
    # Invert targets to simulate the bug
    y_train_inverted = 1 - y_train
    y_test_inverted = 1 - y_test
    
    anfis_broken = AnfisModel(num_mfs_per_input=3)  # Original used 3 MFs
    anfis_broken.train(X_train, y_train_inverted)
    y_pred_broken = anfis_broken.predict(X_test)
    # Predictions are not un-inverted (this was the bug)
    broken_metrics = compute_metrics(y_test, y_pred_broken)  # Compare with original y_test
    
    for metric, value in broken_metrics.items():
        print(f"{metric}: {value:.4f}")
    print()
    
    # Performance comparison
    print("Performance Comparison Summary:")
    print("=" * 40)
    print(f"Linear Regression R²:     {lr_metrics['R2']:.4f}")
    print(f"Fixed ANFIS R²:          {anfis_metrics['R2']:.4f}")
    print(f"Broken ANFIS R² (sim):   {broken_metrics['R2']:.4f}")
    print()
    
    r2_improvement = anfis_metrics['R2'] - broken_metrics['R2']
    print(f"R² Improvement: {r2_improvement:+.4f}")
    
    if anfis_metrics['R2'] > lr_metrics['R2']:
        print("✅ ANFIS outperforms linear regression!")
    elif anfis_metrics['R2'] > broken_metrics['R2']:
        print("✅ Fixed ANFIS is significantly better than broken version!")
    else:
        print("⚠️  Further optimization needed")
    
    # Detailed analysis
    print("\nDetailed Analysis:")
    print("-" * 20)
    print(f"Prediction ranges:")
    print(f"  True values:    [{y_test.min():.3f}, {y_test.max():.3f}]")
    print(f"  Linear Reg:     [{y_pred_lr.min():.3f}, {y_pred_lr.max():.3f}]")
    print(f"  Fixed ANFIS:    [{y_pred_anfis.min():.3f}, {y_pred_anfis.max():.3f}]")
    print(f"  Broken ANFIS:   [{y_pred_broken.min():.3f}, {y_pred_broken.max():.3f}]")

if __name__ == "__main__":
    try:
        test_original_vs_fixed()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
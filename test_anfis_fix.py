#!/usr/bin/env python3
"""
Test script to verify ANFIS fixes improve performance
"""
import numpy as np
import pandas as pd
from models.anfis_model import AnfisModel
from utils.metrics import compute_metrics

def generate_test_data(n_samples=100):
    """Generate synthetic test data for ANFIS validation"""
    np.random.seed(42)
    
    # Generate normalized input features [0, 1]
    TP_norm = np.random.uniform(0, 1, n_samples)
    EC_norm = np.random.uniform(0, 1, n_samples)
    DO_norm = np.random.uniform(0, 1, n_samples)
    tit_norm = np.random.uniform(0, 1, n_samples)
    
    # Create a known relationship for testing
    # y = 0.3*TP + 0.4*EC + 0.2*DO + 0.1*tit + noise
    y = (0.3 * TP_norm + 0.4 * EC_norm + 0.2 * DO_norm + 0.1 * tit_norm + 
         0.05 * np.random.normal(0, 1, n_samples))
    
    # Ensure target is also normalized [0, 1]
    y = np.clip(y, 0, 1)
    
    X = np.column_stack([TP_norm, EC_norm, DO_norm, tit_norm])
    
    return X, y

def test_anfis_performance():
    """Test ANFIS with the fixes"""
    print("Testing ANFIS with fixes...")
    print("=" * 50)
    
    # Generate test data
    X_train, y_train = generate_test_data(200)
    X_test, y_test = generate_test_data(50)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test target range: [{y_test.min():.3f}, {y_test.max():.3f}]")
    print()
    
    # Initialize ANFIS model with new settings
    anfis = AnfisModel(num_mfs_per_input=5)
    
    # Train the model
    print("Training ANFIS model...")
    anfis.train(X_train, y_train)
    print("Training completed!")
    print()
    
    # Make predictions
    print("Making predictions...")
    y_pred = anfis.predict(X_test)
    
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Predictions range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    print()
    
    # Evaluate performance
    print("Performance Metrics:")
    print("-" * 30)
    metrics = compute_metrics(y_test, y_pred)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print()
    print("Expected improvements:")
    print("- R² should be > 0.7 (was ~0.20)")
    print("- RMSE should be < 0.15")
    print("- MAE should be < 0.10")
    
    # Performance assessment
    r2_score = metrics['R2']
    if r2_score > 0.7:
        print(f"\n✅ SUCCESS: R² = {r2_score:.4f} > 0.7 (Much better than original 0.20!)")
    elif r2_score > 0.5:
        print(f"\n⚠️  IMPROVEMENT: R² = {r2_score:.4f} > 0.5 (Better than original 0.20)")
    else:
        print(f"\n❌ ISSUE: R² = {r2_score:.4f} still low (investigate further)")
    
    return metrics

if __name__ == "__main__":
    try:
        metrics = test_anfis_performance()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
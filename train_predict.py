import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from anfis import ANFIS

def train_and_predict(train_path, test_path):
    # Load data
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    
    # Prepare training data
    X_train = torch.FloatTensor(train_df[['TP_norm', 'EC_norm', 'DO_norm', 
                                         'tıt_norm', 'FUZZY_norm']].values)
    y_train = torch.FloatTensor(train_df['SINIF'].values)
    
    # Prepare test data
    X_test = torch.FloatTensor(test_df[['TP_norm', 'EC_norm', 'DO_norm', 
                                       'tıt_norm', 'FUZZY_norm']].values)
    
    # Initialize model and training components
    model = ANFIS(n_inputs=5, n_mf=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training loop
    print("Training ANFIS model...")
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    # Create results table
    results_df = pd.DataFrame({
        'İstasyon': test_df['istasyon'],
        'ANFIS Skoru': predictions.numpy()
    })
    
    # Display results
    print("\nPrediction Results:")
    print(results_df.to_string(index=False))
    
    return results_df

if __name__ == "__main__":
    train_path = "d:/ANFIS_proje/training_data.xlsx"
    test_path = "d:/ANFIS_proje/test_data.xlsx"
    
    results = train_and_predict(train_path, test_path)
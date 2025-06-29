import torch
import torch.nn as nn
import numpy as np
from utils.metrics import compute_metrics
from utils.plots import plot_scatter, plot_residuals
import itertools

class GaussianMembershipFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(1))
        self.sigma = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)

class ANFIS(nn.Module):
    def __init__(self, n_inputs=5, n_mf=2):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_mf = n_mf
        self.n_rules = n_mf ** n_inputs
        
        # Create membership functions for each input
        self.mf_layers = nn.ModuleList([
            nn.ModuleList([GaussianMembershipFunction() for _ in range(n_mf)])
            for _ in range(n_inputs)
        ])

        # Initialize membership function parameters to cover the input space [0, 1]
        # Better initialization: overlapping Gaussian MFs with data-adaptive sigmas
        for i in range(self.n_inputs):
            mus = torch.linspace(0.0, 1.0, self.n_mf)
            # Adaptive sigma based on number of MFs for better coverage
            sigma_val = 1.0 / (2 * self.n_mf)  # Ensures good overlap
            sigmas = torch.full((self.n_mf,), sigma_val)

            for j in range(self.n_mf):
                self.mf_layers[i][j].mu.data = mus[j]
                self.mf_layers[i][j].sigma.data = sigmas[j]
        
        # Consequent parameters for Sugeno rules (not trainable by optimizer directly)
        self.consequent = nn.Parameter(torch.randn(self.n_rules, n_inputs + 1), requires_grad=False)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Layer 1: Membership degrees
        mf_outputs = []
        for i in range(self.n_inputs):
            mf_out = []
            for mf in self.mf_layers[i]:
                mf_out.append(mf(x[:, i:i+1]))
            mf_outputs.append(torch.cat(mf_out, dim=1))
        
        # Layer 2: Rule firing strengths
        mf_values_per_input = [mf_output for mf_output in mf_outputs]

        rule_mf_indices = list(itertools.product(range(self.n_mf), repeat=self.n_inputs))

        firing_strengths = torch.zeros(batch_size, self.n_rules, device=x.device)

        for rule_idx, mf_idx_combination in enumerate(rule_mf_indices):
            rule_strength = torch.ones(batch_size, 1, device=x.device)
            for input_idx, mf_local_idx in enumerate(mf_idx_combination):
                rule_strength *= mf_values_per_input[input_idx][:, mf_local_idx].unsqueeze(1)
            firing_strengths[:, rule_idx] = rule_strength.squeeze()

        # Layer 3: Normalize firing strengths
        sum_firing = firing_strengths.sum(dim=1, keepdim=True)
        # Improved numerical stability
        eps = 1e-8
        normalized_firing = firing_strengths / (sum_firing + eps)
        
        # Layer 4: Rule outputs (for LSE)
        x_aug = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)
        
        # Calculate the output based on current consequent parameters
        output = torch.sum(normalized_firing * torch.matmul(x_aug, self.consequent.t()), dim=1)
        
        return output

class AnfisModel:
    def __init__(self, num_mfs_per_input=5, **kwargs):
        self.model = None
        self.is_trained = False
        self.num_mfs_per_input = num_mfs_per_input
        self.kwargs = kwargs
    
    def train(self, X_train, y_train):
        """ANFIS modelini eğitir (Hybrid Learning)"""
        input_size = X_train.shape[1]
        self.model = ANFIS(n_inputs=input_size, n_mf=self.num_mfs_per_input, **self.kwargs)
        
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1) # Ensure y_train is 2D for MSELoss
        
        # Premise parameters are optimized by Adam with stable learning rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        epochs = 200
        
        for epoch in range(epochs):
            # Step 1: Calculate membership degrees, firing strengths, and normalized firing strengths
            batch_size = X_train_tensor.shape[0]
            mf_outputs = []
            for i in range(self.model.n_inputs):
                mf_out = []
                for mf in self.model.mf_layers[i]:
                    mf_out.append(mf(X_train_tensor[:, i:i+1]))
                mf_outputs.append(torch.cat(mf_out, dim=1))

            mf_values_per_input = [mf_output for mf_output in mf_outputs]
            rule_mf_indices = list(itertools.product(range(self.model.n_mf), repeat=self.model.n_inputs))
            firing_strengths = torch.zeros(batch_size, self.model.n_rules, device=X_train_tensor.device)

            for rule_idx, mf_idx_combination in enumerate(rule_mf_indices):
                rule_strength = torch.ones(batch_size, 1, device=X_train_tensor.device)
                for input_idx, mf_local_idx in enumerate(mf_idx_combination):
                    rule_strength *= mf_values_per_input[input_idx][:, mf_local_idx].unsqueeze(1)
                firing_strengths[:, rule_idx] = rule_strength.squeeze()

            sum_firing = firing_strengths.sum(dim=1, keepdim=True)
            eps = 1e-8
            normalized_firing = firing_strengths / (sum_firing + eps)
            
            x_aug = torch.cat([X_train_tensor, torch.ones(batch_size, 1, device=X_train_tensor.device)], dim=1)

            # Step 2: Least Squares Estimation (LSE) for Consequent Parameters
            # Correct ANFIS LSE: A matrix should be [batch_size, n_rules * (n_inputs + 1)]
            A = torch.zeros(batch_size, self.model.n_rules * (input_size + 1), device=X_train_tensor.device)
            
            for i in range(self.model.n_rules):
                start_idx = i * (input_size + 1)
                end_idx = (i + 1) * (input_size + 1)
                A[:, start_idx:end_idx] = normalized_firing[:, i:i+1] * x_aug
            
            # Solve the linear system using pseudoinverse with regularization
            try:
                # Add small regularization to prevent overfitting
                ATA = torch.matmul(A.t(), A)
                reg_term = 1e-6 * torch.eye(ATA.size(0), device=X_train_tensor.device)
                ATb = torch.matmul(A.t(), y_train_tensor.squeeze())
                consequent_params_flat = torch.linalg.solve(ATA + reg_term, ATb)
                self.model.consequent.data = consequent_params_flat.reshape(self.model.n_rules, input_size + 1)
            except RuntimeError:
                # Fallback to regularized pseudoinverse
                A_reg = torch.cat([A, 1e-3 * torch.eye(A.size(1), device=X_train_tensor.device)], dim=0)
                y_reg = torch.cat([y_train_tensor.squeeze(), torch.zeros(A.size(1), device=X_train_tensor.device)], dim=0)
                consequent_params_flat = torch.linalg.lstsq(A_reg, y_reg).solution
                self.model.consequent.data = consequent_params_flat.reshape(self.model.n_rules, input_size + 1)

            # Step 3: Backward pass for Premise Parameters (using Adam)
            optimizer.zero_grad()
            output_final = self.model(X_train_tensor) # Now only returns the final output
            loss = nn.MSELoss()(output_final, y_train_tensor)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        self.is_trained = True
        # TODO: Implement hyperparameter tuning and early stopping for better performance.
    
    def predict(self, X_test):
        """Tahmin yapar"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        X_test_tensor = torch.FloatTensor(X_test)
        self.model.eval()
        with torch.no_grad():
            # The forward pass now returns (output, normalized_firing, x_aug)
            # We only need the output for prediction
            predictions = self.model.forward(X_test_tensor)
        return predictions.flatten().numpy() # Ensure predictions are 1D numpy array for metrics and plotting
    
    def evaluate(self, y_true, y_pred):
        """Model performansını değerlendirir"""
        return compute_metrics(y_true, y_pred)
    
    def plot_results(self, X_test, y_true, y_pred, output_dir):
        """Sonuçları görselleştirir ve PDF olarak kaydeder"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        plot_scatter(y_true, y_pred, f"{output_dir}/anfis_scatter.pdf", 
                    title="ANFIS: Predicted vs Actual")
        
        plot_residuals(y_true, y_pred, f"{output_dir}/anfis_residuals.pdf") 
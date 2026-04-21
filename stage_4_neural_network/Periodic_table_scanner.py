"""
Periodic Table Sweep for RBS Surrogate Models
Loads trained neural networks (energy loss and backscatter probability)
and predicts for all elements Z = 1 ... 92 at a given thickness and energy.
Plots the results to verify physical trends.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# define model architecture
class SmallNN(nn.Module):
    def __init__(self, hidden_neurons=8):
        super(SmallNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_neurons),   # inputs: Z, thickness (Å), E0 (MeV)
            nn.ReLU(),
            nn.Linear(hidden_neurons, 1)
        )
    def forward(self, x):
        return self.net(x)

#hlper function
def predict_sweep(model_path, Z_list, thickness_A, E0_MeV):

    # Load checkpoint with weights_only=False
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = SmallNN(hidden_neurons=8)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    X_mean = checkpoint['X_mean']
    X_std  = checkpoint['X_std']
    y_mean = checkpoint['y_mean']
    y_std  = checkpoint['y_std']
    log_target = checkpoint.get('log_target', False)

    predictions = []
    for Z in Z_list:
        X = np.array([[Z, thickness_A, E0_MeV]], dtype=np.float32)
        X_norm = (X - X_mean) / X_std
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        with torch.no_grad():
            y_norm = model(X_tensor).numpy()[0, 0]
            y = y_norm * y_std + y_mean
            if log_target:
                y = 10 ** y
        predictions.append(y)
    return np.array(predictions)

# main function
if __name__ == "__main__":
    # Setting
    Z_values = np.arange(1, 93)          # Hydrogen (1) to Uranium (92)
    thickness_A = 5000.0                 # thickness in Ångströms
    E0_MeV = 2.20                        # incident energy in MeV

    # Paths to saved models 
    model_dir = "Analysis_Results"
    loss_model_path = os.path.join(model_dir, "NN_model_Energy_Loss_(MeV).pt")
    back_model_path = os.path.join(model_dir, "NN_model_Backscatter_Probability.pt")

    # Check files exist
    if not os.path.exists(loss_model_path):
        raise FileNotFoundError(f"Energy loss model not found at {loss_model_path}")
    if not os.path.exists(back_model_path):
        raise FileNotFoundError(f"Backscatter model not found at {back_model_path}")

    # Run predictions
    print("Running Periodic Table sweep...")
    pred_loss = predict_sweep(loss_model_path, Z_values, thickness_A, E0_MeV)
    pred_back = predict_sweep(back_model_path, Z_values, thickness_A, E0_MeV)
    print("Done.")

    # Print warning about extrapolation
    print("Note: Training Z range is 6 (carbon) to 79 (gold). Predictions for Z < 6 and Z > 79 are extrapolations.")

    # plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Energy loss (linear scale)
    ax1.plot(Z_values, pred_loss, 'b-', linewidth=2)
    # Mark training limits
    ax1.axvline(x=6, color='gray', linestyle='--', alpha=0.7, label='Training limits (Z=6 to 79)')
    ax1.axvline(x=79, color='gray', linestyle='--', alpha=0.7)
    # Shade extrapolation regions
    ax1.axvspan(1, 6, alpha=0.2, color='red', label='Extrapolation')
    ax1.axvspan(79, 92, alpha=0.2, color='red')
    ax1.set_xlabel("Atomic number Z")
    ax1.set_ylabel("Predicted energy loss (MeV)")
    ax1.set_title(f"Energy loss sweep (thickness = {thickness_A:.0f} Å, E₀ = {E0_MeV} MeV)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Backscatter probability (log scale)
    ax2.semilogy(Z_values, pred_back, 'r-', linewidth=2)
    ax2.axvline(x=6, color='gray', linestyle='--', alpha=0.7, label='Training limits (Z=6 to 79)')
    ax2.axvline(x=79, color='gray', linestyle='--', alpha=0.7)
    ax2.axvspan(1, 6, alpha=0.2, color='red', label='Extrapolation')
    ax2.axvspan(79, 92, alpha=0.2, color='red')
    ax2.set_xlabel("Atomic number Z")
    ax2.set_ylabel("Backscatter probability")
    ax2.set_title("Backscatter probability sweep (log scale)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    # Save figure
    output_plot = "Z_sweep_validation.png"
    plt.savefig(output_plot, dpi=150)
    print(f"Plot saved as {output_plot}")
    plt.show()
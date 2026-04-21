"""
RBS Surrogate Model Predictor
Loads trained neural networks and predicts energy loss, backscatter probability,
and mean scattering angle from Z, thickness (Å), and incident energy (MeV).
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# model setup
class SmallNN(nn.Module):
    def __init__(self, hidden_neurons=8):
        super(SmallNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, 1)
        )
    def forward(self, x):
        return self.net(x)

# helper
def load_model_and_predict(model_path, X_input):
    """
    Load a saved PyTorch model and normalization parameters,
    then predict for a single input X_input (numpy array of shape (1,3)).
    Returns the predicted value in original units.
    """
    # weights_only=False because checkpoint contains numpy arrays
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = SmallNN(hidden_neurons=8)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    X_mean = checkpoint['X_mean']
    X_std = checkpoint['X_std']
    y_mean = checkpoint['y_mean']
    y_std = checkpoint['y_std']
    log_target = checkpoint.get('log_target', False)

    # Normalize input
    X_norm = (X_input - X_mean) / X_std
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    with torch.no_grad():
        y_norm = model(X_tensor).numpy()[0, 0]
        y = y_norm * y_std + y_mean
        if log_target:
            y = 10 ** y
    return y

# main loop
def main():
    # Path to models 
    base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, "models")

    loss_path = os.path.join(model_dir, "NN_model_Energy_Loss_(MeV).pt")
    back_path = os.path.join(model_dir, "NN_model_Backscatter_Probability.pt")
    angle_path = os.path.join(model_dir, "NN_model_Mean_Scattering_Angle_(deg).pt")

    # Check that all model files exist
    for path in [loss_path, back_path, angle_path]:
        if not os.path.exists(path):
            print(f"ERROR: Model file not found: {path}")
            print("Please ensure the .pt files are in the 'models' folder next to the executable.")
            input("Press Enter to exit...")
            sys.exit(1)

    print("\n" + "="*50)
    print("RBS Surrogate Model Predictor")
    print("="*50)
    print("Enter parameters to predict:\n")

    while True:
        try:
            Z = int(input("Atomic number Z (1-92): "))
            thickness_A = float(input("Thickness (Å): "))
            E0_MeV = float(input("Incident energy (MeV): "))
        except ValueError:
            print("Invalid input. Please enter numbers.\n")
            continue

        # Prepare input array
        X = np.array([[Z, thickness_A, E0_MeV]], dtype=np.float32)

        # Predict
        try:
            energy_loss = load_model_and_predict(loss_path, X)
            back_prob = load_model_and_predict(back_path, X)
            mean_angle = load_model_and_predict(angle_path, X)
        except Exception as e:
            print(f"Prediction failed: {e}\n")
            continue

        # Display results
        print("\nResults")
        print(f"Energy loss         : {energy_loss:.4f} MeV")
        print(f"Backscatter probability : {back_prob:.4e}")
        print(f"Mean scattering angle   : {mean_angle:.4f}°")
        print("-"*30)

        # Ask for another prediction
        again = input("\nPredict another? (y/n): ").strip().lower()
        if again != 'y':
            break
        print()


if __name__ == "__main__":
    main()
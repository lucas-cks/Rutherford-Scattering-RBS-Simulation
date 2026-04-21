"""
Analyze RBS simulation results from the batch runs.
Aggregates data, computes statistics, produces plots,
and trains neural networks to predict outcomes.

Train a neural network to predict the full 170 deg energy spectrum.
"""

import os
import re
import csv
import glob
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Suppress matplotlib log-scale warning 
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# import torch if available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Neural network prediction will be skipped.")

# Configuration
RESULTS_BASE = "Results"          # where result folders are
OUTPUT_DIR = "Analysis_Results"   # where analysis outputs go
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(RANDOM_SEED)

# Use parallel processing for parsing (set to True if many folders)
USE_PARALLEL = False   # Set to True to use multiple CPU cores
MAX_WORKERS = 8

# Number of bins in the 170° energy spectrum files (should match the simulation output)
ENERGY_BINS = 100

# Helper functions for parsing
def safe_extract(pattern, content, dtype=float, default=None):
    """Extract first matching group from content using regex, with safe fallback.
       Supports scientific notation (e.g., 1.2e-05)."""
    match = re.search(pattern, content, flags=re.DOTALL)
    if match:
        try:
            return dtype(match.group(1))
        except (ValueError, TypeError):
            return default
    return default

def parse_simulation_results(filepath):
    """Extract key values from simulation_results.txt using safe regex."""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Required fields
    Z = safe_extract(r'Atomic number Z = (\d+)', content, int)
    if Z is None:
        return {}
    data['Z'] = Z
    data['density'] = safe_extract(r'Density = ([\d\.]+) g/cm³', content, float, 0.0)
    data['A'] = safe_extract(r'Atomic mass A = ([\d\.]+) g/mol', content, float, 0.0)
    data['n_atom'] = safe_extract(r'Atom density = ([\d\.e\+\-]+) atoms/cm³', content, float, 0.0)
    data['thickness_cm'] = safe_extract(r'Thickness = ([\d\.e\+\-]+) cm', content, float, 0.0)
    data['N_layers'] = safe_extract(r'Number of layers = (\d+)', content, int, 0)
    data['E0_MeV'] = safe_extract(r'Initial kinetic energy = ([\d\.]+) MeV', content, float, 0.0)
    data['N_particles'] = safe_extract(r'Total number of simulated particles: (\d+)', content, int, 0)
    # Support scientific notation for mean and std
    data['mean_final_energy_MeV'] = safe_extract(r'Final Energy \(MeV\).*?Mean = ([\d\.e\+\-]+)', content, float, None)
    data['std_final_energy_MeV'] = safe_extract(r'Final Energy \(MeV\).*?Standard deviation = ([\d\.e\+\-]+)', content, float, None)
    # Scattering angle block (supports scientific notation)
    angle_block = re.search(r'Scattering Angle \(degrees\):.*?Mean = ([\d\.e\+\-]+).*?Standard deviation = ([\d\.e\+\-]+)', content, re.DOTALL)
    if angle_block:
        data['mean_angle_deg'] = float(angle_block.group(1))
        data['std_angle_deg'] = float(angle_block.group(2))
    else:
        data['mean_angle_deg'] = 0.0
        data['std_angle_deg'] = 0.0
    data['backscatter_prob'] = safe_extract(r'Backscattering \(θ>90°\).*?Probability = ([\d\.e\+\-]+)', content, float, 0.0)

    # Compute energy loss 
    if data['E0_MeV'] and data['mean_final_energy_MeV'] is not None:
        data['energy_loss_MeV'] = data['E0_MeV'] - data['mean_final_energy_MeV']
    else:
        data['energy_loss_MeV'] = 0.0

    if data['thickness_cm']:
        data['thickness_A'] = data['thickness_cm'] * 1e8
    else:
        data['thickness_A'] = 0.0

    return data

def parse_energy_spectrum(filepath):
    """
    Read energy_spectrum_170.csv and return (energy_bins_center, weighted_counts).
    Assumes the file has exactly ENERGY_BINS bins.
    """
    energies = []
    counts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                energies.append(float(parts[0]))
                counts.append(float(parts[1]))
    # If the number of bins is not exactly ENERGY_BINS, we can either truncate or pad with zeros
    if len(counts) != ENERGY_BINS:
        print(f"Warning: {filepath} has {len(counts)} bins, expected {ENERGY_BINS}. Truncating/padding.")
        if len(counts) > ENERGY_BINS:
            counts = counts[:ENERGY_BINS]
            energies = energies[:ENERGY_BINS]
        else:
            counts += [0.0] * (ENERGY_BINS - len(counts))
            energies += [energies[-1] + (energies[1]-energies[0])] * (ENERGY_BINS - len(counts))
    return np.array(energies), np.array(counts)

def parse_histogram(filepath):
    """Read histogram.csv and return (angle_center, prob_density)."""
    angles = []
    prob = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                angles.append(float(parts[0]))
                prob.append(float(parts[1]))
    return np.array(angles), np.array(prob)

def parse_single_folder(folder):
    """Parse one result folder and return (data_dict, spec_data, hist_data)."""
    sim_file = os.path.join(folder, "simulation_results.txt")
    if not os.path.exists(sim_file):
        return None, None, None
    data = parse_simulation_results(sim_file)
    if not data:
        return None, None, None
    data['folder'] = os.path.basename(folder)

    # Energy spectrum
    spec_file = os.path.join(folder, "energy_spectrum_170.csv")
    spec_data = None
    if os.path.exists(spec_file):
        E_center, counts = parse_energy_spectrum(spec_file)
        # Only keep spectra that have non‑zero counts (otherwise they are empty)
        if np.sum(counts) > 0:
            spec_data = (data['Z'], data['thickness_cm'], data['E0_MeV'], E_center, counts)

    # Histogram
    hist_file = os.path.join(folder, "histogram.csv")
    hist_data = None
    if os.path.exists(hist_file):
        ang_center, prob = parse_histogram(hist_file)
        hist_data = (data['Z'], data['thickness_cm'], data['E0_MeV'], ang_center, prob)

    return data, spec_data, hist_data

# NN Definitions
# Smaller network for scalar regression (avoid overfitting with small data)
class SmallNN(nn.Module):
    def __init__(self, input_dim=3, hidden_neurons=8, output_dim=1):
        super(SmallNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class SpectrumNN(nn.Module):
    """Neural network to predict a full energy spectrum (multi-output regression)."""
    def __init__(self, input_dim=3, hidden_layers=[128, 256, 128], output_dim=ENERGY_BINS):
        super(SpectrumNN, self).__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def robust_normalize(X, X_mean=None, X_std=None):
    """Normalize features, handling near‑zero standard deviation."""
    if X_mean is None:
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        # If std is extremely small (constant feature), set it to 1.0 to avoid division by zero
        X_std = np.where(X_std < 1e-12, 1.0, X_std)
        X_norm = (X - X_mean) / X_std
        return X_norm, X_mean, X_std
    else:
        X_std_adj = np.where(X_std < 1e-12, 1.0, X_std)
        X_norm = (X - X_mean) / X_std_adj
        return X_norm

def train_scalar_network(X, y, output_name, log_target=False, hidden_neurons=8, save_model=True):
    """
    Train a small neural network for scalar outputs.
    If log_target=True, transform y = log10(y + 1e-12) before training.
    """
    # Filter out non-finite values
    mask = np.isfinite(y)
    if not np.all(mask):
        print(f"  Removing {np.sum(~mask)} non‑finite y values for {output_name}")
        X = X[mask]
        y = y[mask]

    if log_target:
        # Ensure positive values for log
        mask_pos = y > 1e-12
        if not np.any(mask_pos):
            print(f"Warning: No positive y for {output_name}. Training will fail.")
            return None, None
        X = X[mask_pos]
        y = y[mask_pos]
        y = np.log10(y + 1e-12)
        print(f"  Using log10 transformation for {output_name} (kept {len(y)} samples)")
    else:
        print(f"  Training {output_name} with {len(y)} samples (no log transform)")

    # Normalize features
    X_norm, X_mean, X_std = robust_normalize(X)
    y_mean = y.mean()
    y_std = y.std()
    y_norm = (y - y_mean) / (y_std + 1e-12)

    print(f"  Training stats: X_mean={X_mean}, X_std={X_std}, y_mean={y_mean:.4e}, y_std={y_std:.4e}")

    # Train/test split (80/20)
    n = len(X)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = torch.tensor(X_norm[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_norm[train_idx], dtype=torch.float32).view(-1,1)
    X_test = torch.tensor(X_norm[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y_norm[test_idx], dtype=torch.float32).view(-1,1)

    # Build small model
    model = SmallNN(input_dim=X.shape[1], hidden_neurons=hidden_neurons, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    epochs = 500
    batch_size = 32
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_test).numpy().flatten()
        y_pred = y_pred_norm * y_std + y_mean
        y_true = y_test.numpy().flatten() * y_std + y_mean

        if log_target:
            y_pred_lin = 10 ** y_pred
            y_true_lin = 10 ** y_true
            mse = np.mean((y_true_lin - y_pred_lin)**2)
            ss_res = np.sum((y_true_lin - y_pred_lin)**2)
            ss_tot = np.sum((y_true_lin - np.mean(y_true_lin))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            print(f"\nNeural network for {output_name} (log10 transformed):")
            print(f"  Test MSE (linear): {mse:.4e}")
            print(f"  Test R² (linear): {r2:.4f}")
        else:
            mse = np.mean((y_true - y_pred)**2)
            r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
            print(f"\nNeural network for {output_name}:")
            print(f"  Test MSE: {mse:.4e}")
            print(f"  Test R² : {r2:.4f}")

    # Plot predictions vs true
    plt.figure(figsize=(6,6))
    if log_target:
        y_true_plot = 10 ** y_true
        y_pred_plot = 10 ** y_pred
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
    plt.scatter(y_true_plot, y_pred_plot, alpha=0.6)
    lims = [min(y_true_plot.min(), y_pred_plot.min()), max(y_true_plot.max(), y_pred_plot.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.title(f'Neural network prediction: {output_name}')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'NN_prediction_{output_name.replace(" ", "_")}.png'), dpi=150)
    plt.close()

    # Save model and normalization parameters
    if save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std,
            'log_target': log_target
        }, os.path.join(OUTPUT_DIR, f'NN_model_{output_name.replace(" ", "_")}.pt'))
        print(f"  Model saved to {OUTPUT_DIR}/NN_model_{output_name.replace(' ', '_')}.pt")

    return model, (X_mean, X_std, y_mean, y_std, log_target)

def train_spectrum_network(X, y_spectra, energy_bins, save_model=True):
    """
    Train a neural network to predict the full energy spectrum.
    X: input features (n_samples, n_features)
    y_spectra: list or array of shape (n_samples, ENERGY_BINS)
    """
    # Add a tiny offset to avoid zero counts (log transform would fail)
    y_spectra = y_spectra + 1e-8

    # Normalize inputs
    X_norm, X_mean, X_std = robust_normalize(X)

    # Normalize outputs: each bin separately
    y_mean = y_spectra.mean(axis=0)
    y_std = y_spectra.std(axis=0)
    y_std = np.where(y_std < 1e-12, 1.0, y_std)
    y_norm = (y_spectra - y_mean) / y_std

    # Train/test split
    n = len(X)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = torch.tensor(X_norm[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_norm[train_idx], dtype=torch.float32)
    X_test = torch.tensor(X_norm[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y_norm[test_idx], dtype=torch.float32)

    # Build model
    model = SpectrumNN(input_dim=X.shape[1], hidden_layers=[128, 256, 128], output_dim=ENERGY_BINS)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    epochs = 800
    batch_size = 32
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_test).numpy()
        y_pred = y_pred_norm * y_std + y_mean
        y_true = y_test.numpy() * y_std + y_mean
        mse_per_bin = np.mean((y_true - y_pred)**2, axis=0)
        overall_mse = np.mean(mse_per_bin)
        print(f"\nSpectrum prediction network:")
        print(f"  Test overall MSE: {overall_mse:.4e}")
        print(f"  Mean MSE per bin: {np.mean(mse_per_bin):.4e}")

    # Plot a few example predictions
    n_examples = min(5, len(X_test))
    fig, axes = plt.subplots(1, n_examples, figsize=(5*n_examples, 4))
    if n_examples == 1:
        axes = [axes]
    for i in range(n_examples):
        axes[i].plot(energy_bins, y_true[i], 'b-', label='True')
        axes[i].plot(energy_bins, y_pred[i], 'r--', label='Predicted')
        axes[i].set_xlabel('Energy (MeV)')
        axes[i].set_ylabel('Counts')
        axes[i].legend()
        axes[i].set_title(f'Test sample {i+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'spectrum_NN_examples.png'), dpi=150)
    plt.close()

    # Save model and normalization parameters
    if save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std,
            'energy_bins': energy_bins
        }, os.path.join(OUTPUT_DIR, 'NN_model_Energy_Spectrum.pt'))
        print(f"  Model saved to {OUTPUT_DIR}/NN_model_Energy_Spectrum.pt")

    return model, (X_mean, X_std, y_mean, y_std, energy_bins)

# Main analysis function
def main():
    # Find all result folders
    result_folders = glob.glob(os.path.join(RESULTS_BASE, "Result_*"))
    print(f"Found {len(result_folders)} result folders.")

    all_data = []
    spectra_dict = defaultdict(list)   # key: (Z, thickness_cm, E0_MeV) -> list of (E_center, counts)
    angles = defaultdict(list)         # key: (Z, thickness_cm, E0_MeV) -> list of (angle_center, prob)

    if USE_PARALLEL:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_folder = {executor.submit(parse_single_folder, folder): folder for folder in result_folders}
            for future in as_completed(future_to_folder):
                data, spec_data, hist_data = future.result()
                if data is not None:
                    all_data.append(data)
                if spec_data is not None:
                    Z, thick, E0, E_center, counts = spec_data
                    spectra_dict[(Z, thick, E0)].append((E_center, counts))
                if hist_data is not None:
                    Z, thick, E0, ang_center, prob = hist_data
                    angles[(Z, thick, E0)].append((ang_center, prob))
    else:
        for folder in result_folders:
            data, spec_data, hist_data = parse_single_folder(folder)
            if data is not None:
                all_data.append(data)
            if spec_data is not None:
                Z, thick, E0, E_center, counts = spec_data
                spectra_dict[(Z, thick, E0)].append((E_center, counts))
            if hist_data is not None:
                Z, thick, E0, ang_center, prob = hist_data
                angles[(Z, thick, E0)].append((ang_center, prob))

    print(f"Successfully parsed {len(all_data)} simulation runs.")

    # Save all raw data to CSV
    if all_data:
        keys = all_data[0].keys()
        with open(os.path.join(OUTPUT_DIR, "all_results.csv"), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_data)

    # Aggregate scalar quantities by (Z, thickness, E0) – average over seeds (for plotting and spectrum)
    agg = defaultdict(lambda: {'count': 0, 'sum_loss': 0.0, 'sum_back': 0.0,
                                'sum_angle': 0.0, 'sum_angle_std': 0.0})
    for d in all_data:
        key = (d['Z'], d['thickness_cm'], d['E0_MeV'])
        agg[key]['count'] += 1
        agg[key]['sum_loss'] += d.get('energy_loss_MeV', 0)
        agg[key]['sum_back'] += d.get('backscatter_prob', 0)
        agg[key]['sum_angle'] += d.get('mean_angle_deg', 0)
        agg[key]['sum_angle_std'] += d.get('std_angle_deg', 0)

    summary = []
    for (Z, thick, E0), vals in agg.items():
        n = vals['count']
        summary.append({
            'Z': Z,
            'thickness_cm': thick,
            'thickness_A': thick * 1e8,
            'E0_MeV': E0,
            'N_seeds': n,
            'mean_energy_loss_MeV': vals['sum_loss'] / n,
            'mean_backscatter_prob': vals['sum_back'] / n,
            'mean_angle_deg': vals['sum_angle'] / n,
            'mean_angle_std_deg': vals['sum_angle_std'] / n,
        })

    with open(os.path.join(OUTPUT_DIR, "summary_aggregated.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    # Scalar Plots
    # Plot energy loss vs thickness for each material and energy
    materials_to_plot = [79, 78, 13, 14, 6, 8, 32]
    for Z in materials_to_plot:
        subset = [s for s in summary if s['Z'] == Z]
        if not subset:
            continue
        plt.figure(figsize=(8,6))
        energies = sorted(set(s['E0_MeV'] for s in subset))
        for E in energies:
            sub = [s for s in subset if s['E0_MeV'] == E]
            thick_vals = [s['thickness_A'] for s in sub]
            loss_vals = [s['mean_energy_loss_MeV'] for s in sub]
            plt.plot(thick_vals, loss_vals, 'o-', label=f'{E} MeV')
        plt.xlabel('Thickness (Å)')
        plt.ylabel('Energy loss (MeV)')
        plt.title(f'Energy loss vs thickness for Z={Z}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, f'energy_loss_Z{Z}.png'), dpi=150)
        plt.close()

    Z = 79
    subset = [s for s in summary if s['Z'] == Z]
    if subset:
        plt.figure(figsize=(8,6))
        thicknesses = sorted(set(s['thickness_A'] for s in subset))
        for thick in thicknesses:
            sub = [s for s in subset if s['thickness_A'] == thick]
            E_vals = [s['E0_MeV'] for s in sub]
            prob_vals = [s['mean_backscatter_prob'] for s in sub]
            plt.plot(E_vals, prob_vals, 'o-', label=f'{thick:.0f} Å')
        plt.xlabel('Incident energy (MeV)')
        plt.ylabel('Backscatter probability')
        plt.title(f'Backscatter probability vs energy for Z={Z}')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(os.path.join(OUTPUT_DIR, f'backscatter_prob_Z{Z}.png'), dpi=150)
        plt.close()

    # Spectra and angular plots 
    for (Z, thick, E0), spec_list in spectra_dict.items():
        if len(spec_list) < 2:
            continue
        first_E, _ = spec_list[0]
        avg_counts = np.zeros_like(first_E)
        for E, cnt in spec_list:
            if len(E) != len(first_E):
                continue
            avg_counts += cnt
        avg_counts /= len(spec_list)
        plt.figure(figsize=(8,6))
        plt.plot(first_E, avg_counts, 'b-')
        plt.xlabel('Energy (MeV)')
        plt.ylabel('Weighted counts (arb. units)')
        plt.title(f'170° backscattered spectrum\nZ={Z}, thick={thick*1e8:.0f}Å, E0={E0}MeV')
        plt.grid(True)
        filename = f'spectrum_Z{Z}_thick{thick*1e8:.0f}Ang_E{E0:.2f}MeV.png'
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
        plt.close()

    for (Z, thick, E0), ang_list in angles.items():
        if len(ang_list) < 2:
            continue
        first_ang, _ = ang_list[0]
        avg_prob = np.zeros_like(first_ang)
        for ang, prob in ang_list:
            if len(ang) != len(first_ang):
                continue
            avg_prob += prob
        avg_prob /= len(ang_list)
        plt.figure(figsize=(8,6))
        plt.plot(first_ang, avg_prob, 'r-')
        plt.xlabel('Scattering angle (degrees)')
        plt.ylabel('Probability density (1/deg)')
        plt.title(f'Angular distribution\nZ={Z}, thick={thick*1e8:.0f}Å, E0={E0}MeV')
        if np.min(avg_prob) > 0:
            plt.yscale('log')
        plt.grid(True)
        filename = f'angular_Z{Z}_thick{thick*1e8:.0f}Ang_E{E0:.2f}MeV.png'
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
        plt.close()

    # NN training
    if not TORCH_AVAILABLE:
        print("\nPyTorch not installed. Skipping neural network training.")
        print("Install with: pip install torch")
        return

    # Scalar models: train on raw data
    # Extract features and targets from individual runs (2100 samples)
    X_raw = []
    y_loss_raw = []
    y_back_raw = []
    y_angle_raw = []
    for d in all_data:
        X_raw.append([d['Z'], d['thickness_A'], d['E0_MeV']])
        y_loss_raw.append(d['energy_loss_MeV'])
        y_back_raw.append(d['backscatter_prob'])
        y_angle_raw.append(d['mean_angle_deg'])
    X_raw = np.array(X_raw)
    y_loss_raw = np.array(y_loss_raw)
    y_back_raw = np.array(y_back_raw)
    y_angle_raw = np.array(y_angle_raw)

    print("\nScalar data stats:")
    print(f"  Number of samples: {len(X_raw)}")
    print(f"  Energy loss: min={np.min(y_loss_raw):.4e}, max={np.max(y_loss_raw):.4e}, mean={np.mean(y_loss_raw):.4e}")
    print(f"  Backscatter prob: min={np.min(y_back_raw):.4e}, max={np.max(y_back_raw):.4e}, mean={np.mean(y_back_raw):.4e}")
    print(f"  Mean angle: min={np.min(y_angle_raw):.4f}, max={np.max(y_angle_raw):.4f}, mean={np.mean(y_angle_raw):.4f}")

    print("\n--- Training scalar neural networks on raw data")
    # Energy loss
    model_loss, norm_loss = train_scalar_network(X_raw, y_loss_raw, "Energy Loss (MeV)", log_target=False, hidden_neurons=8)
    # Backscatter probability (with log transform)
    model_back, norm_back = train_scalar_network(X_raw, y_back_raw, "Backscatter Probability", log_target=True, hidden_neurons=8)
    # Mean scattering angle
    model_angle, norm_angle = train_scalar_network(X_raw, y_angle_raw, "Mean Scattering Angle (deg)", log_target=False, hidden_neurons=8)

    # Spectrum network
    spectrum_inputs = []
    spectrum_targets = []
    spectrum_energy_bins = None
    z_counts = defaultdict(int)
    for (Z, thick, E0), spec_list in spectra_dict.items():
        if len(spec_list) == 0:
            continue
        first_E, _ = spec_list[0]
        avg_counts = np.zeros_like(first_E)
        valid = 0
        for E, cnt in spec_list:
            if len(E) == ENERGY_BINS:
                avg_counts += cnt
                valid += 1
        if valid == 0:
            continue
        avg_counts /= valid
        spectrum_inputs.append([Z, thick*1e8, E0])
        spectrum_targets.append(avg_counts)
        spectrum_energy_bins = first_E
        z_counts[Z] += 1

    spectrum_inputs = np.array(spectrum_inputs)
    spectrum_targets = np.array(spectrum_targets)
    print(f"\nPrepared spectrum dataset: {len(spectrum_inputs)} samples")
    print("Samples per Z:")
    for z, cnt in sorted(z_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  Z={z}: {cnt} samples")

    if len(spectrum_inputs) > 10:
        print("--- Training spectrum neural network ---")
        model_spectrum, norm_spectrum = train_spectrum_network(spectrum_inputs, spectrum_targets, spectrum_energy_bins, save_model=True)

        # Example prediction
        if len(z_counts) > 0:
            most_common_z = max(z_counts.items(), key=lambda x: x[1])[0]
            existing = [inp for inp in spectrum_inputs if inp[0] == most_common_z][0]
            print(f"\nExample spectrum prediction for Z={int(existing[0])}, thickness={existing[1]:.0f} Å, E0={existing[2]:.2f} MeV (existing in training)")
            new_X = np.array([existing])
            X_mean_spec, X_std_spec, y_mean_spec, y_std_spec, _ = norm_spectrum
            new_X_norm = (new_X - X_mean_spec) / X_std_spec
            new_X_tensor = torch.tensor(new_X_norm, dtype=torch.float32)
            model_spectrum.eval()
            with torch.no_grad():
                pred_norm = model_spectrum(new_X_tensor).numpy().flatten()
                pred_counts = pred_norm * y_std_spec + y_mean_spec
            plt.figure(figsize=(8,6))
            plt.plot(spectrum_energy_bins, pred_counts, 'b-', label='Predicted spectrum')
            plt.xlabel('Energy (MeV)')
            plt.ylabel('Weighted counts')
            plt.title(f'Predicted 170° backscattered spectrum (Z={int(existing[0])})')
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, 'predicted_spectrum_example.png'), dpi=150)
            plt.close()
            print("  Predicted spectrum plot saved.")
    else:
        print("Not enough spectrum samples to train neural network. Skip.")

    # Scalar prediction example
    if model_loss is not None:
        print("\nExample scalar predictions for a configuration present in training")
        # Pick the first sample from raw data
        example = X_raw[0]
        print(f"  Using Z={int(example[0])}, thickness={example[1]:.0f} Å, E0={example[2]:.2f} MeV")

        # Energy loss prediction
        X_mean_loss, X_std_loss, y_mean_loss, y_std_loss, log_used_loss = norm_loss
        new_X = np.array([example])
        new_X_norm = (new_X - X_mean_loss) / X_std_loss
        new_X_tensor = torch.tensor(new_X_norm, dtype=torch.float32)
        model_loss.eval()
        with torch.no_grad():
            pred_norm = model_loss(new_X_tensor).numpy().flatten()[0]
            pred_loss = pred_norm * y_std_loss + y_mean_loss
        print(f"  Predicted energy loss: {pred_loss:.4f} MeV")

        # Backscatter probability prediction
        X_mean_back, X_std_back, y_mean_back, y_std_back, log_used_back = norm_back
        new_X_norm_back = (new_X - X_mean_back) / X_std_back
        new_X_tensor_back = torch.tensor(new_X_norm_back, dtype=torch.float32)
        model_back.eval()
        with torch.no_grad():
            pred_norm_back = model_back(new_X_tensor_back).numpy().flatten()[0]
            pred_log_back = pred_norm_back * y_std_back + y_mean_back
            if log_used_back:
                pred_back = 10 ** pred_log_back
            else:
                pred_back = pred_log_back
        print(f"  Predicted backscatter probability: {pred_back:.4e}")

        # Mean angle prediction
        X_mean_angle, X_std_angle, y_mean_angle, y_std_angle, _ = norm_angle
        new_X_norm_angle = (new_X - X_mean_angle) / X_std_angle
        new_X_tensor_angle = torch.tensor(new_X_norm_angle, dtype=torch.float32)
        model_angle.eval()
        with torch.no_grad():
            pred_norm_angle = model_angle(new_X_tensor_angle).numpy().flatten()[0]
            pred_angle = pred_norm_angle * y_std_angle + y_mean_angle
        print(f"  Predicted mean scattering angle: {pred_angle:.4f}°")

    print("\nAll analysis completed.")

if __name__ == "__main__":
    main()

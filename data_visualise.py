import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.stats import norm

# ============================================
# 1. Configuration & Input Filenames
# ============================================
L = 10.0               # Distance from foil to detector screen (cm)
ZOOM = 2.0             # Heatmap display range [-ZOOM, ZOOM] (cm)
BINS_2D = 200          # Resolution for Heatmap
BINS_ANGLE = 100       # Bins for Angular Distribution
BINS_ENERGY = 100      # Bins for Energy Spectrum

txt_filename = "simulation_results.txt"  # Your C-generated report
csv_filename = "results_raw.csv"         # Your C-generated raw data

# Physical Constants (SI Units)
E_CHARGE = 1.602176634e-19
EPSILON0 = 8.8541878128e-12

# ============================================
# 2. Automated Parameter Extraction (Regex)
# ============================================
def extract_parameters(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_path} not found. Please run the C simulation first.")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Using Regular Expressions to capture physical values from the TXT report
    params = {
        'Z': int(re.search(r'Atomic number Z = (\d+)', content).group(1)),
        'n_atom': float(re.search(r'Atom density = ([\d.e+-]+)', content).group(1)),
        'z_inc': int(re.search(r'Charge number z = (\d+)', content).group(1)),
        'Ek_MeV': float(re.search(r'Initial kinetic energy = ([\d.e+-]+)', content).group(1)),
        'thickness': float(re.search(r'Thickness = ([\d.e+-]+)', content).group(1)),
        'total_n': int(re.search(r'Total number of particles simulated: (\d+)', content).group(1)),
        'backscatter': int(re.search(r'Count = (\d+)', content).group(1))
    }
    return params

print("Step 1: Extracting physical parameters from report...")
p = extract_parameters(txt_filename)

# ============================================
# 3. Dynamic Folder Setup
# ============================================
# Create a unique folder based on simulation parameters to avoid overwriting data
folder_name = f"Result_Z{p['Z']}_T{p['thickness']:.2e}_N{p['total_n']}_E{p['Ek_MeV']:.2f}MeV"
os.makedirs(folder_name, exist_ok=True)
print(f"Step 2: Results will be saved in directory: '{folder_name}'")

# ============================================
# 4. Data Loading & Statistical Analysis
# ============================================
print(f"Step 3: Reading raw data from {csv_filename}...")
df = pd.read_csv(csv_filename, on_bad_lines='skip')
total_rows = len(df)

if total_rows == 0:
    print("Error: CSV file is empty. Exiting.")
    exit()

mean_e = df['final_energy_MeV'].mean()
std_e = df['final_energy_MeV'].std()
mean_a = df['scattering_angle_deg'].mean()
std_a = df['scattering_angle_deg'].std()
back_prob = p['backscatter'] / p['total_n']


print("\n" + "="*50)
print("SIMULATION SUMMARY STATS")
print(f"Final Energy (MeV): Mean = {mean_e:.4f}, Std Dev = {std_e:.4f}")
print(f"Scattering Angle (deg): Mean = {mean_a:.4f}, Std Dev = {std_a:.4f}")
print(f"Backscattering Count: {p['backscatter']} / {p['total_n']} (Prob: {back_prob:.6f})")
print("="*50 + "\n")

# ============================================
# 5. Plot 1: Particle Impact Heatmap (2D)
# ============================================
print("Step 4: Generating 2D Heatmap...")
# Filter only forward-moving particles to project onto the screen
forward = df[df['dir_x'] > 0].copy()
if len(forward) > 0:
    # Project 3D vector onto a 2D screen at distance L
    t = L / forward['dir_x']
    y_screen = t * forward['dir_y']
    z_screen = t * forward['dir_z']

    plt.figure(figsize=(8, 8))
    plt.hist2d(y_screen, z_screen, bins=BINS_2D, range=[[-ZOOM, ZOOM], [-ZOOM, ZOOM]],
               cmap='hot', cmin=1)
    plt.colorbar(label='Hit Count')
    plt.xlabel('y-coordinate on screen (cm)')
    plt.ylabel('z-coordinate on screen (cm)')
    plt.title(f'Detector Screen Impact Map (Distance L = {L} cm)\nTarget Z={p["Z"]}, Energy={p["Ek_MeV"]} MeV')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "heatmap.png"), dpi=300)
    plt.close()

# ============================================
# 6. Plot 2: Angular Distribution vs Absolute Theory
# ============================================
print("Step 5: Simulating Result VS Theoretical Result")
# Calculate Absolute Theoretical Rutherford Cross-Section
Ek_joule = p['Ek_MeV'] * 1e6 * E_CHARGE
# Rutherford Constant in m^2/sr
constant_si = ((p['z_inc'] * p['Z'] * E_CHARGE**2) / (16 * np.pi * EPSILON0 * Ek_joule))**2
constant_cm2 = constant_si * 1e4 # Convert m^2 to cm^2

# Setup Angular Array (Avoid theta = 0 due to singularity)
theta_deg = np.linspace(1.0, 179.0, 500)
theta_rad = np.radians(theta_deg)
ds_domega = constant_cm2 / (np.sin(theta_rad/2)**4)

# Absolute Probability Density (1/deg) = (n * t) * (dσ/dΩ) * (2π sinθ) * (π/180)
# This represents the expected fraction of particles per degree
theory_pdf = (p['n_atom'] * p['thickness']) * ds_domega * (2 * np.pi * np.sin(theta_rad)) * (np.pi/180)

plt.figure(figsize=(10, 6))
plt.hist(df['scattering_angle_deg'], bins=BINS_ANGLE, density=True, alpha=0.6, 
         label='Monte Carlo Simulation', color='steelblue')
plt.plot(theta_deg, theory_pdf, 'r-', linewidth=2, label='Rutherford csc^4(θ/2) Absolute Theory')

plt.yscale('log')
plt.xlabel('Scattering Angle (degrees)')
plt.ylabel('Probability Density (1/deg)')
plt.title(f'Absolute Comparison: Simulation vs Theory\nTarget Z={p["Z"]}, Thickness={p["thickness"]:.2e} cm')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(folder_name, "angle_distribution.png"), dpi=300)
plt.close()

# ============================================
# 7. Plot 3: Energy Spectrum (Straggling Effect)
# ============================================
print("Step 6: Generating Energy Spectrum with Gaussian Fit...")
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(df['final_energy_MeV'], bins=BINS_ENERGY, 
                            color='forestgreen', edgecolor='black', alpha=0.7, 
                            density=True, label='Simulated Spectrum')

# Fit Gaussian to quantify Energy Straggling (Bohr Theory)
mu, std = norm.fit(df['final_energy_MeV'])
x_fit = np.linspace(min(bins), max(bins), 100)
plt.plot(x_fit, norm.pdf(x_fit, mu, std), 'r--', linewidth=2, 
         label=f'Gaussian Fit ($\sigma$={std:.4f} MeV)')

plt.xlabel('Final Kinetic Energy (MeV)')
plt.ylabel('Probability Density')
plt.title(f'Energy Spectrum: Straggling Analysis\nInitial Energy: {p["Ek_MeV"]} MeV, Mean Loss: {p["Ek_MeV"] - mu:.3f} MeV')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(folder_name, "energy_spectrum.png"), dpi=300)
plt.close()

print(f"\nSUCCESS: All figures saved to '{folder_name}/'")
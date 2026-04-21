#!/usr/bin/env python3
"""
Batch driver for RBS Monte Carlo simulation (CSV mode).
Reads run_list.csv, materials.csv, particles.csv.
Runs the C executable and saves results into per-run folders.
"""

import csv
import subprocess
import os
import shutil
import sys

# Configuration
EXECUTABLE = "./RBS_openmp.exe"          
OUTPUT_BASE = "Results"           

def load_materials(filename='materials.csv'):
    """
    Return dict {material_id: (Z, density, A, a0, a1, a2, a3, a4)}
    Also return dict {Z: material_id} for lookup.
    """
    materials_by_id = {}
    id_by_Z = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)   # skip header
        for row in reader:
            if len(row) < 10:
                continue
            try:
                mat_id = int(row[0].strip())
                Z = int(row[2].strip())
                den = float(row[3].strip())
                A = float(row[4].strip())
                a0 = float(row[5].strip())
                a1 = float(row[6].strip())
                a2 = float(row[7].strip())
                a3 = float(row[8].strip())
                a4 = float(row[9].strip())
                materials_by_id[mat_id] = (Z, den, A, a0, a1, a2, a3, a4)
                id_by_Z[Z] = mat_id
            except (ValueError, IndexError) as e:
                print(f"Warning: skipping row {row} due to error: {e}")
                continue
    return materials_by_id, id_by_Z

def load_particles(filename='particles.csv'):
    """
    Return dict {particle_id: (z, mass_MeV, name)}
    """
    particles = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 5:
                continue
            pid = int(row[0].strip())
            z = int(row[2].strip())
            mass_MeV = float(row[4].strip())
            name = row[1].strip()
            particles[pid] = (z, mass_MeV, name)
    return particles

def run_simulation(mat_id, part_id, thickness, N_layers, seed, N_particles):
    """
    Launch one simulation using CSV mode (1).
    Returns True if successful.
    """
    # Build input string for CSV mode
    input_str = (
        f"1\n"                # select CSV mode
        f"{mat_id}\n"         # material number
        f"{part_id}\n"        # particle number
        f"{thickness}\n"      # thickness (cm)
        f"{N_layers}\n"       # number of layers
        f"{N_particles}\n"    # number of particles (asked in main)
        f"{seed}\n"           # random seed
    )

    try:
        # Use DEVNULL to ignore stdout/stderr and avoid UnicodeDecodeError
        proc = subprocess.run(EXECUTABLE, input=input_str, text=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=3600)
        if proc.returncode != 0:
            print(f"Simulation failed (returncode {proc.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print("Simulation timed out (>1 hour)")
        return False
    except Exception as e:
        print(f"Error launching subprocess: {e}")
        return False
    return True

def create_result_folder(Z, thickness, N_layers, energy_MeV, N_particles, seed):
    """Create a unique folder name with seed to avoid collisions."""
    thick_str = f"{thickness:.2e}".replace('e-0', 'e-').replace('e+0', 'e+')
    energy_str = f"{energy_MeV:.2f}MeV"
    folder_name = f"Result_Z{Z}_T{thick_str}_N{N_particles}_E{energy_str}_seed{seed}"
    folder_path = os.path.join(OUTPUT_BASE, folder_name)
    os.makedirs(folder_path, exist_ok=False)   # fail if exists
    return folder_path

def move_outputs(dest_folder):
    """Move expected output files into dest_folder (keep original names)."""
    output_files = [
        "histogram.csv",
        "simulation_results.txt",
        "results_raw.csv",
        "energy_spectrum_170.csv"
    ]
    for fname in output_files:
        if os.path.exists(fname):
            shutil.move(fname, os.path.join(dest_folder, fname))

def is_run_completed(Z, thickness, N_layers, energy_MeV, N_particles, seed):
    """Check if simulation_results.txt already exists for this run."""
    thick_str = f"{thickness:.2e}".replace('e-0', 'e-').replace('e+0', 'e+')
    energy_str = f"{energy_MeV:.2f}MeV"
    folder_name = f"Result_Z{Z}_T{thick_str}_N{N_particles}_E{energy_str}_seed{seed}"
    folder_path = os.path.join(OUTPUT_BASE, folder_name)
    result_file = os.path.join(folder_path, "simulation_results.txt")
    return os.path.exists(result_file)

def main():
    if not os.path.isfile(EXECUTABLE):
        print(f"Error: Executable '{EXECUTABLE}' not found in current directory.")
        sys.exit(1)

    materials_by_id, id_by_Z = load_materials()
    particles = load_particles()

    # Read run_list.csv
    run_list = []
    with open("run_list.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 8:
                continue
            Z = int(row[0].strip())
            part_id = int(row[1].strip())
            thickness = float(row[2].strip())
            N_layers = int(row[3].strip())
            seed = int(row[4].strip())
            energy_MeV = float(row[6].strip())
            N_particles = int(row[7].strip())
            if Z not in id_by_Z:
                print(f"Error: Z={Z} not found in materials.csv")
                continue
            mat_id = id_by_Z[Z]
            run_list.append((mat_id, part_id, thickness, N_layers, seed, energy_MeV, N_particles, Z))

    print(f"Loaded {len(run_list)} simulation runs.")

    for idx, (mat_id, part_id, thickness, N_layers, seed, energy_MeV, N_parts, Z) in enumerate(run_list, 1):
        # Skip if already completed
        if is_run_completed(Z, thickness, N_layers, energy_MeV, N_parts, seed):
            print(f"[{idx}/{len(run_list)}] Skipping (already completed): Z={Z}, thick={thickness:.2e}, E={energy_MeV:.3f} MeV, seed={seed}")
            continue

        print(f"[{idx}/{len(run_list)}] Z={Z}, part={part_id}, "
              f"thick={thickness:.2e} cm, layers={N_layers}, "
              f"E={energy_MeV:.3f} MeV, N={N_parts}, seed={seed}")

        success = run_simulation(mat_id, part_id, thickness, N_layers, seed, N_parts)
        if not success:
            print("  Simulation failed – skipping.")
            continue

        # Create folder (allow existing) and move outputs
        thick_str = f"{thickness:.2e}".replace('e-0', 'e-').replace('e+0', 'e+')
        energy_str = f"{energy_MeV:.2f}MeV"
        folder_name = f"Result_Z{Z}_T{thick_str}_N{N_parts}_E{energy_str}_seed{seed}"
        folder_path = os.path.join(OUTPUT_BASE, folder_name)
        os.makedirs(folder_path, exist_ok=True)   # create if not exists
        move_outputs(folder_path)
        print(f"  Results saved to: {folder_path}")

    print("All done.")

if __name__ == "__main__":
    main()

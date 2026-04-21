#!/usr/bin/env python3
import csv

# This script generates run_list.csv for the RBS Monte Carlo simulation.
Z_list = [79, 78, 13, 14, 6, 8, 32]   # Au, Pt, Al, Si, C, O, Ge

# incident particle ID (1 for alpha particles in particles.csv)
particle_id = 1

# parameter combinations: (thickness_cm, N_layers, energy_MeV)
param_combos = [
    (4e-5, 100, 1.54),
    (4e-5, 100, 3.30),
    (4e-5, 100, 2.20),
    (4e-5, 100, 2.72),
    (4e-5, 100, 4.00),
    (3e-4, 200, 1.54),
    (3e-4, 200, 3.30),
    (3e-4, 200, 2.20),
    (3e-4, 200, 2.72),
    (3e-4, 200, 4.00),
    (8e-5, 100, 1.54),
    (8e-5, 100, 3.30),
    (8e-5, 100, 2.20),
    (8e-5, 100, 2.72),
    (8e-5, 100, 4.00),
    (1e-4, 200, 1.54),
    (1e-4, 200, 3.30),
    (1e-4, 200, 2.20),
    (1e-4, 200, 2.72),
    (1e-4, 200, 4.00),
    (2e-4, 400, 1.54),
    (2e-4, 400, 3.30),
    (2e-4, 400, 2.20),
    (2e-4, 400, 2.72),
    (2e-4, 400, 4.00),
    (5e-4, 1000, 1.54),
    (5e-4, 1000, 3.30),
    (5e-4, 1000, 2.20),
    (5e-4, 1000, 2.72),
    (5e-4, 1000, 4.00),
]

seeds = list(range(1, 11))          # seeds 1..10
N_particles_fixed = 1000000

with open('run_list.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # 新 header: Z, particle_id, thickness_cm, N_layers, seed, unused, energy_MeV, N_particles
    writer.writerow(['Z', 'particle_id', 'thickness_cm', 'N_layers',
                     'seed', 'unused', 'energy_MeV', 'N_particles'])
    for Z in Z_list:
        for thick, layers, energy in param_combos:
            for seed in seeds:
                writer.writerow([Z, particle_id, thick, layers, seed,
                                 0, f"{energy:.6e}", N_particles_fixed])

print(f"Generated run_list.csv with {len(Z_list)*len(param_combos)*len(seeds)} lines.")
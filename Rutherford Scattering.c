// Rutherford Scattering

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Some Constants
const double pi = 3.141592653589793;
const double speed_light = 299792458.0;   // m/s
const double m_e = 9.10938356e-31;        // kg
const double N_A = 6.02214076e23;         // mol^-1
const double e_charge = 1.602176634e-19;  // C
const double K_SI = 7.342522e-25;         // J·m^4/s^2
const double epsilon_0 = 8.854187817e-12; // F/m

// Variables (material, particle)
int Z;                  // atomic number of target
double den;             // density (g/cm^3)
double A;               // atomic mass (g/mol)
double n_atom;          // atoms/cm^3
double n_electron;      // electrons/cm^3
double thickness;       // target thickness (cm)
int N_layers;           // number of layers
double dx;              // layer thickness (cm)

int z;                  // projectile charge number
double I;               // mean excitation energy (eV)
double v;               // current velocity (m/s)
double mass_incident;   // projectile mass (kg)

double pos[3] = {0.0, 0.0, 0.0};
double dir[3] = {1.0, 0.0, 0.0};

double initial_energy;          // initial kinetic energy (J)
double initial_v;               // initial velocity (m/s)
double initial_dir[3];
double initial_energy_save;

double final_energy;
int final_dimensionless_position;

double sum_energy = 0.0, sum_energy2 = 0.0;
double sum_angle = 0.0, sum_angle2 = 0.0;
int backscatter_count = 0;

double total_areal_density;     // atoms/cm^2

// Histogram for AI training
#define HIST_BINS 100
double hist[HIST_BINS] = {0.0};
double bin_width;

// MT19937 PRNG 
#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfUL
#define MT_UPPER_MASK 0x80000000UL
#define MT_LOWER_MASK 0x7fffffffUL

static uint32_t mt[MT_N];
static int mti = MT_N + 1;

void init_genrand(uint32_t s) {
    mt[0] = s & 0xffffffffUL;
    for (mti = 1; mti < MT_N; mti++) {
        mt[mti] = (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
        mt[mti] &= 0xffffffffUL;
    }
}

uint32_t genrand_int32(void) {
    uint32_t y;
    static uint32_t mag01[2] = {0x0UL, MT_MATRIX_A};
    int kk;

    if (mti >= MT_N) {
        for (kk = 0; kk < MT_N - MT_M; kk++) {
            y = (mt[kk] & MT_UPPER_MASK) | (mt[kk+1] & MT_LOWER_MASK);
            mt[kk] = mt[kk+MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < MT_N - 1; kk++) {
            y = (mt[kk] & MT_UPPER_MASK) | (mt[kk+1] & MT_LOWER_MASK);
            mt[kk] = mt[kk+(MT_M-MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[MT_N-1] & MT_UPPER_MASK) | (mt[0] & MT_LOWER_MASK);
        mt[MT_N-1] = mt[MT_M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];
        mti = 0;
    }

    y = mt[mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
}

double ranf(void) {
    return genrand_int32() * (1.0 / 4294967296.0);
}

// Bethe‑Bloch stopping power
double bethe_bloch(double v) {
    double n_e_m3 = n_electron * 1e6;   // cm⁻³ → m⁻³
    if (v < 0.01 * speed_light) {
        // non‑relativistic
        double I_J = I * e_charge;
        double k = (K_SI * n_e_m3 * z * z) / (v * v);
        double inside = (2.0 * m_e * v * v) / I_J;
        if (inside <= 1e-20) inside = 1e-20;
        return k * log(inside);
    } else {
        // relativistic
        double beta = v / speed_light;
        double beta_sq = beta * beta;
        double I_J = I * e_charge;
        double k = (K_SI * n_e_m3 * z * z) / (v * v);
        double denom = I_J * fmax(1.0 - beta_sq, 1e-12);
        double inside = (2.0 * m_e * v * v) / denom;
        if (inside <= 1e-20) inside = 1e-20;
        return k * (log(inside) - beta_sq);
    }
}

// Energy loss
double energy_loss() {
    double n_atom_m3 = n_atom * 1e6;
    double dx_m = dx * 0.01;

    double coulomb_const = e_charge * e_charge / (4.0 * pi * epsilon_0);
    double sigma_sq = 4.0 * pi * z * z * coulomb_const * coulomb_const * Z * n_atom_m3 * dx_m;
    double sigma = sqrt(fmax(0.0, sigma_sq));

    double r1 = ranf(), r2 = ranf();
    double gaussian_noise = sqrt(-2.0 * log(fmax(r1, 1e-16))) * cos(2.0 * pi * r2);

    double energy_loss_J = bethe_bloch(v) * dx_m + gaussian_noise * sigma;
    if (energy_loss_J < 0.0) energy_loss_J = 0.0;

    final_energy -= energy_loss_J;
    if (final_energy < 0.0) final_energy = 0.0;
    final_dimensionless_position += 1;

    return final_energy;
}

void new_velocity() {
    double total_energy_J = final_energy + mass_incident * speed_light * speed_light;
    if (v < 0.01 * speed_light) {
        v = sqrt(2.0 * final_energy / mass_incident);
    } else {
        double gamma = total_energy_J / (mass_incident * speed_light * speed_light);
        v = speed_light * sqrt(1.0 - 1.0 / (gamma * gamma));
    }
    if (v <= 0.0) {
        printf("Particle stopped at position: %d layers\n", (int)final_dimensionless_position);
    }
}

// Scattering 
int scattering_determine(double b) {
    double cross_section = pi * b * b;           // geometric cross section (m^2)
    double prob = cross_section * n_atom * dx;   // probability per layer
    return (ranf() < prob) ? 1 : 0;
}

void yes_new_position_and_direction(double b, double E_scatter_J, double xi, double theta, double phi) {
    // Move to scattering point
    for (int j = 0; j < 3; j++) pos[j] += xi * dx * dir[j];
    total_areal_density += n_atom * (xi * dx);

    // Rotate direction
    double new_dir[3];
    rotate_direction(dir, theta, phi, new_dir);

    // Move remaining distance
    for (int j = 0; j < 3; j++) pos[j] += (1.0 - xi) * dx * new_dir[j];
    total_areal_density += n_atom * ((1.0 - xi) * dx);

    // Update direction
    for (int j = 0; j < 3; j++) dir[j] = new_dir[j];
}

void rotate_direction(double dir[3], double theta, double phi, double new_dir[3]) {
    double u[3], v[3];
    double wx = dir[0], wy = dir[1], wz = dir[2];
    double norm_w = sqrt(wx*wx + wy*wy + wz*wz);
    if (norm_w > 1e-12) { wx /= norm_w; wy /= norm_w; wz /= norm_w; }

    if (fabs(wx) < 0.9) {
        u[0] = 0.0; u[1] = -wz; u[2] = wy;
    } else {
        u[0] = wz; u[1] = 0.0; u[2] = -wx;
    }
    double norm_u = sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
    if (norm_u > 1e-12) { u[0] /= norm_u; u[1] /= norm_u; u[2] /= norm_u; }
    else { u[0] = 1.0; u[1] = 0.0; u[2] = 0.0; }

    v[0] = wy * u[2] - wz * u[1];
    v[1] = wz * u[0] - wx * u[2];
    v[2] = wx * u[1] - wy * u[0];
    double norm_v = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (norm_v > 1e-12) { v[0] /= norm_v; v[1] /= norm_v; v[2] /= norm_v; }

    double sin_theta = sin(theta), cos_theta = cos(theta);
    double sin_phi = sin(phi), cos_phi = cos(phi);

    new_dir[0] = sin_theta * cos_phi * u[0] + sin_theta * sin_phi * v[0] + cos_theta * wx;
    new_dir[1] = sin_theta * cos_phi * u[1] + sin_theta * sin_phi * v[1] + cos_theta * wy;
    new_dir[2] = sin_theta * cos_phi * u[2] + sin_theta * sin_phi * v[2] + cos_theta * wz;

    double norm = sqrt(new_dir[0]*new_dir[0] + new_dir[1]*new_dir[1] + new_dir[2]*new_dir[2]);
    if (norm > 1e-12) { new_dir[0] /= norm; new_dir[1] /= norm; new_dir[2] /= norm; }
}

void no_new_position_and_direction() {
    for (int j = 0; j < 3; j++) pos[j] += dx * dir[j];
    total_areal_density += n_atom * dx;
}

// Main Loop
void main_loop() {
    n_atom = N_A * den / A;
    n_electron = Z * n_atom;
    dx = thickness / N_layers;
    final_energy = initial_energy;
    final_dimensionless_position = 0;

    double b_max = pow(1.0 / n_atom, 1.0/3.0) / 2.0;   // maximum impact parameter (cm)
    double b = b_max;

    for (int i = 0; i < N_layers; i++) {
        if (scattering_determine(b)) {
            // Scattering occurs in this layer
            double u = ranf();
            double b_actual = b * sqrt(u);                 // sample b uniformly in area

            double xi = ranf();                            // position of scatter inside layer

            // Energy loss before scattering
            double dx_m = dx * 0.01;
            double dEdx = bethe_bloch(v);
            double energy_loss_to_scatter = dEdx * (xi * dx_m);
            double E_scatter_J = final_energy - energy_loss_to_scatter;
            if (E_scatter_J < 0) E_scatter_J = 0;

            // Rutherford angle
            double b_m = b_actual * 0.01;                  // cm → m
            double theta = 2.0 * atan( (double)z * Z * e_charge * e_charge /
                                       (8.0 * pi * epsilon_0 * E_scatter_J * b_m) );
            double phi = 2.0 * pi * ranf();

            yes_new_position_and_direction(b_actual, E_scatter_J, xi, theta, phi);

            // Energy loss after scattering
            final_energy = energy_loss();
            new_velocity();
        } else {
            // No scattering
            no_new_position_and_direction();
            final_energy = energy_loss();
            new_velocity();
        }
        if (v <= 0) break;
    }
}

// Input handling 
void input_parameters() {
    int choice;
    printf("Select input mode:\n1. Load from csv files (materials.csv and particles.csv)\n2. Manual input\nChoice: ");
    scanf("%d", &choice);

    if (choice == 1) {
        // 1. Load material
        FILE *file = fopen("materials.csv", "r");
        if (!file) { printf("Error: Could not open materials.csv\n"); exit(1); }
        char buffer[256];
        fgets(buffer, sizeof(buffer), file);
        printf("\nAvailable materials:\n");
        while (fgets(buffer, sizeof(buffer), file)) {
            int num; char name[50];
            double Zmat, denval, Aval, Ival;
            if (sscanf(buffer, "%d, %49[^,], %lf, %lf, %lf, %lf",
                       &num, name, &Zmat, &denval, &Aval, &Ival) == 6) {
                printf("%d: %s (Z=%.0f, ρ=%.2f g/cm^3, A=%.2f g/mol, I=%.0f eV)\n",
                       num, name, Zmat, denval, Aval, Ival);
            }
        }
        fclose(file);
        int mat_choice;
        printf("\nEnter the number of the material you want: ");
        scanf("%d", &mat_choice);

        file = fopen("materials.csv", "r");
        if (!file) { printf("Error: Could not reopen materials.csv\n"); exit(1); }
        fgets(buffer, sizeof(buffer), file);
        int found = 0;
        while (fgets(buffer, sizeof(buffer), file)) {
            int num; char name[50];
            double Zmat, denval, Aval, Ival;
            if (sscanf(buffer, "%d, %49[^,], %lf, %lf, %lf, %lf",
                       &num, name, &Zmat, &denval, &Aval, &Ival) == 6) {
                if (num == mat_choice) {
                    Z = (int)Zmat;
                    den = denval;
                    A = Aval;
                    I = Ival;
                    printf("Selected material: %s\n", name);
                    found = 1;
                    break;
                }
            }
        }
        fclose(file);
        if (!found) { printf("Error: Material number %d not found.\n", mat_choice); exit(1); }

        // 2. Load particle
        file = fopen("particles.csv", "r");
        if (!file) { printf("Error: Could not open particles.csv\n"); exit(1); }
        fgets(buffer, sizeof(buffer), file);
        printf("\nAvailable particles:\n");
        while (fgets(buffer, sizeof(buffer), file)) {
            int num; char name[50]; int zval; double vval, mval;
            if (sscanf(buffer, "%d, %49[^,], %d, %lf, %lf",
                       &num, name, &zval, &vval, &mval) == 5) {
                printf("%d: %s (z=%d, v=%.2e m/s, mass=%.2e MeV/c²)\n",
                       num, name, zval, vval, mval);
            }
        }
        fclose(file);
        int part_choice;
        printf("\nEnter the number of the particle you want: ");
        scanf("%d", &part_choice);

        file = fopen("particles.csv", "r");
        if (!file) { printf("Error: Could not reopen particles.csv\n"); exit(1); }
        fgets(buffer, sizeof(buffer), file);
        found = 0;
        while (fgets(buffer, sizeof(buffer), file)) {
            int num; char name[50]; int zval; double vval, mval;
            if (sscanf(buffer, "%d, %49[^,], %d, %lf, %lf",
                       &num, name, &zval, &vval, &mval) == 5) {
                if (num == part_choice) {
                    z = zval;
                    v = vval;
                    mass_incident = mval;
                    printf("Selected particle: %s\n", name);
                    found = 1;
                    break;
                }
            }
        }
        fclose(file);
        if (!found) { printf("Error: Particle number %d not found.\n", part_choice); exit(1); }

        // 3. Geometry
        printf("\nEnter thickness (cm) of the material (Original Geiger-Marsden Experiment as reference:0.00006cm): ");
        scanf("%lf", &thickness);
        printf("Enter number of layers (foils) in the simulation (Original Geiger-Marsden Experiment as reference:1000): ");
        scanf("%d", &N_layers);
    } else {
        // Manual input
        printf("Enter atomic number (Z) of the material: ");
        scanf("%d", &Z);
        printf("Enter density (g/cm^3) of the material: ");
        scanf("%lf", &den);
        printf("Enter atomic mass (g/mol) of the material: ");
        scanf("%lf", &A);
        printf("Enter thickness (cm) of the material: ");
        scanf("%lf", &thickness);
        printf("Enter number of layers (foils) in the simulation: ");
        scanf("%d", &N_layers);
        printf("Enter charge number (z) of the incident particle: ");
        scanf("%d", &z);
        printf("Enter mean excitation energy (I in eV) of the material: ");
        scanf("%lf", &I);
        printf("Enter velocity (m/s) of the incident particle: ");
        scanf("%lf", &v);
        printf("Enter rest mass (MeV/c^2) of the incident particle: ");
        scanf("%lf", &mass_incident);
    }
    mass_incident = mass_incident * 1.78266192e-30; // MeV/c² → kg
    initial_v = v;
}




// Output
void output_histogram(const char *hist_filename, long long N_particles,
                      double hist[], int bins, double bin_width_deg) {
    FILE *f = fopen(hist_filename, "w");
    if (!f) { printf("Error: Could not open %s\n", hist_filename); return; }
    fprintf(f, "# Histogram of scattering angles (degrees)\n");
    fprintf(f, "# bins=%d, range=[0,180], bin_width=%.2f\n", bins, bin_width_deg);
    fprintf(f, "# Total particles: %lld\n", N_particles);
    for (int i = 0; i < bins; i++) {
        double angle_center = (i + 0.5) * bin_width_deg;
        fprintf(f, "%.4f %.6f\n", angle_center, hist[i]);
    }
    fclose(f);
}

void output_results(const char *filename,
                    long long N_particles,
                    double mean_E, double var_E,
                    double mean_angle, double var_angle,
                    int backscatter_count) {
    FILE *outfile = fopen(filename, "w");
    if (!outfile) { printf("Error: Could not open output file %s\n", filename); return; }

    fprintf(outfile, "Rutherford Scattering Simulation Results\n");
    fprintf(outfile, "========================================\n\n");
    fprintf(outfile, "Input Parameters:\n");
    fprintf(outfile, "-----------------\n");
    fprintf(outfile, "Material:\n");
    fprintf(outfile, "  Atomic number Z = %d\n", Z);
    fprintf(outfile, "  Density = %.2f g/cm³\n", den);
    fprintf(outfile, "  Atomic mass A = %.2f g/mol\n", A);
    fprintf(outfile, "  Mean excitation energy I = %.0f eV\n", I);
    fprintf(outfile, "  Atom density = %.2e atoms/cm³\n", n_atom);
    fprintf(outfile, "  Electron density = %.2e electrons/cm³\n\n", n_electron);
    fprintf(outfile, "Geometry:\n");
    fprintf(outfile, "  Thickness = %.10f cm\n", thickness);
    fprintf(outfile, "  Number of layers = %d\n", N_layers);
    fprintf(outfile, "  Layer thickness dx = %.2e cm\n\n", dx);
    fprintf(outfile, "Incident Particle:\n");
    fprintf(outfile, "  Charge number z = %d\n", z);
    fprintf(outfile, "  Initial velocity = %.2e m/s\n", initial_v);
    fprintf(outfile, "  Rest mass = %.2e kg\n", mass_incident);
    fprintf(outfile, "  Initial kinetic energy = %.2f MeV\n", initial_energy_save / 1.60218e-13);
    fprintf(outfile, "  Initial direction = (%.2f, %.2f, %.2f)\n\n", initial_dir[0], initial_dir[1], initial_dir[2]);
    fprintf(outfile, "Simulation Results:\n");
    fprintf(outfile, "----------------------------------------------\n");
    fprintf(outfile, "Total number of particles simulated: %lld\n", N_particles);
    fprintf(outfile, "Final Energy (MeV):\n");
    fprintf(outfile, "  Mean = %.4f\n", mean_E);
    fprintf(outfile, "  Variance = %.4f\n", var_E);
    fprintf(outfile, "  Standard deviation = %.4f\n", sqrt(var_E));
    fprintf(outfile, "\nScattering Angle (degrees):\n");
    fprintf(outfile, "  Mean = %.4f\n", mean_angle);
    fprintf(outfile, "  Variance = %.4f\n", var_angle);
    fprintf(outfile, "  Standard deviation = %.4f\n", sqrt(var_angle));
    fprintf(outfile, "\nBackscattering:\n");
    fprintf(outfile, "  Count = %d\n", backscatter_count);
    if (backscatter_count > 0) {
        fprintf(outfile, "  Probability = %.6f (1/%.0f)\n",
                (double)backscatter_count / N_particles,
                (double)N_particles / backscatter_count);
    } else {
        fprintf(outfile, "  Probability = 0.000000 (no backscattering)\n");
    }
    fclose(outfile);
    printf("Results saved to %s\n", filename);
}


// Main
int main() {
    input_parameters();

    initial_dir[0] = dir[0];
    initial_dir[1] = dir[1];
    initial_dir[2] = dir[2];

    // Initial kinetic energy (relativistic if needed)
    if (initial_v < 0.01 * speed_light) {
        initial_energy = 0.5 * mass_incident * initial_v * initial_v;
    } else {
        double beta = initial_v / speed_light;
        double gamma = 1.0 / sqrt(1.0 - beta * beta);
        initial_energy = (gamma - 1.0) * mass_incident * speed_light * speed_light;
    }
    initial_energy_save = initial_energy;

    long long N_particles;
    printf("Enter number of particles to simulate: ");
    scanf("%lld", &N_particles);
    printf("Simulating %lld particles\n", N_particles);

    init_genrand(5489UL);   // default seed

    // Initialize histogram
    bin_width = 180.0 / HIST_BINS;
    for (int i = 0; i < HIST_BINS; i++) hist[i] = 0.0;

    FILE *raw = fopen("results_raw.csv", "w");
    if (raw) {
        fprintf(raw, "particle_id,final_energy_MeV,scattering_angle_deg,backscattered,dir_x,dir_y,dir_z,total_areal_density\n");
    }

    for (long long i = 0; i < N_particles; i++) {
        printf("\nSimulating particle %lld\n", i+1);
        fflush(stdout);

        total_areal_density = 0.0;
        pos[0] = pos[1] = pos[2] = 0.0;
        dir[0] = 1.0; dir[1] = 0.0; dir[2] = 0.0;
        final_energy = initial_energy;
        final_dimensionless_position = 0;
        v = initial_v;

        main_loop();

        double E_MeV = final_energy / 1.60218e-13;
        double dot = dir[0];
        if (dot > 1.0) dot = 1.0;
        if (dot < -1.0) dot = -1.0;
        double angle_rad = acos(dot);
        double angle_deg = angle_rad * 180.0 / pi;
        int backscattered = (dir[0] < 0) ? 1 : 0;
        if (backscattered) backscatter_count++;

        sum_energy += E_MeV;
        sum_energy2 += E_MeV * E_MeV;
        sum_angle += angle_deg;
        sum_angle2 += angle_deg * angle_deg;

        // Histogram update
        int bin = (int)(angle_deg / bin_width);
        if (bin >= HIST_BINS) bin = HIST_BINS - 1;
        hist[bin] += 1.0;

        if (raw) {
            fprintf(raw, "%lld,%.4f,%.2f,%d,%.6f,%.6f,%.6f,%.2e\n",
                    i+1, E_MeV, angle_deg, backscattered,
                    dir[0], dir[1], dir[2], total_areal_density);
        }
    }
    if (raw) fclose(raw);

    // Normalize histogram
    double total = 0.0;
    for (int i = 0; i < HIST_BINS; i++) total += hist[i];
    if (total > 0) {
        for (int i = 0; i < HIST_BINS; i++) hist[i] /= total;
        for (int i = 0; i < HIST_BINS; i++) hist[i] /= bin_width;
    }
    output_histogram("histogram.csv", N_particles, hist, HIST_BINS, bin_width);

    double mean_E = sum_energy / N_particles;
    double var_E = (sum_energy2 / N_particles) - (mean_E * mean_E);
    double mean_angle = sum_angle / N_particles;
    double var_angle = (sum_angle2 / N_particles) - (mean_angle * mean_angle);

    output_results("simulation_results.txt",
                   N_particles,
                   mean_E, var_E,
                   mean_angle, var_angle,
                   backscatter_count);

    return 0;
}
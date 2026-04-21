// Rutherford Backsattering 
// with Stable Importance Sampling and OpenMP
// Compile: gcc -o RBS_openmp.exe RBS.c -lm -fopenmp

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Some consatnts
const double pi = 3.141592653589793;
const double speed_light = 299792458.0;   // m/s
const double m_e = 9.10938356e-31;        // kg
const double N_A = 6.02214076e23;         // mol^-1
const double e_charge = 1.602176634e-19;  // C
const double epsilon_0 = 8.854187817e-12; // F/m
const double eV_to_J = 1.602176634e-19;   // 1 eV = 1.602176634e-19 J

// MT19937 constants
#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfUL
#define MT_UPPER_MASK 0x80000000UL
#define MT_LOWER_MASK 0x7fffffffUL

// Per‑thread RNG state
typedef struct {
    uint32_t mt[MT_N];
    int mti;
} mt_state;

// Global variables
int Z;                  
double den;             
double A;               
double n_atom;          
double thickness;       
int N_layers;           
double dx;              

int z;                  
double v;               
double mass_incident;   

double eps_coeff[5]; 

double initial_energy;          
double initial_v;               
double initial_dir[3];
double initial_energy_save;

// detector statistics
double sum_weight = 0.0;        
double sum_energy_w = 0.0, sum_energy2_w = 0.0;
double sum_angle_w = 0.0, sum_angle2_w = 0.0;
double backscatter_weight = 0.0;

#define HIST_BINS 100
#define ENERGY_BINS_170 100

double hist[HIST_BINS] = {0.0};
double energy_hist_170[ENERGY_BINS_170] = {0.0};

double bin_width;
double E_max_bins;  // max energy for 17 deg° histogram

//==================================================================
// MT19937 functions
void init_genrand_thread(mt_state* state, uint32_t s) {
    state->mt[0] = s & 0xffffffffUL;
    for (state->mti = 1; state->mti < MT_N; state->mti++) {
        state->mt[state->mti] = (1812433253UL * (state->mt[state->mti-1] ^ (state->mt[state->mti-1] >> 30)) + state->mti);
        state->mt[state->mti] &= 0xffffffffUL;
    }
}

uint32_t genrand_int32_thread(mt_state* state) {
    uint32_t y;
    static uint32_t mag01[2] = {0x0UL, MT_MATRIX_A};
    int kk;

    if (state->mti >= MT_N) {
        for (kk = 0; kk < MT_N - MT_M; kk++) {
            y = (state->mt[kk] & MT_UPPER_MASK) | (state->mt[kk+1] & MT_LOWER_MASK);
            state->mt[kk] = state->mt[kk+MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < MT_N - 1; kk++) {
            y = (state->mt[kk] & MT_UPPER_MASK) | (state->mt[kk+1] & MT_LOWER_MASK);
            state->mt[kk] = state->mt[kk+(MT_M-MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (state->mt[MT_N-1] & MT_UPPER_MASK) | (state->mt[0] & MT_LOWER_MASK);
        state->mt[MT_N-1] = state->mt[MT_M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];
        state->mti = 0;
    }

    y = state->mt[state->mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
}

double ranf_thread(mt_state* state) {
    return genrand_int32_thread(state) * (1.0 / 4294967296.0);
}


// Stopping power and energy loss
double stopping_cross_section(double E_MeV) {
    double e = E_MeV;
    return eps_coeff[0] + eps_coeff[1]*e + eps_coeff[2]*e*e +
           eps_coeff[3]*e*e*e + eps_coeff[4]*e*e*e*e;
}

double dEdx_J_per_m(double E_MeV) {
    double epsilon = stopping_cross_section(E_MeV);  
    double dEdx_eV_per_cm = n_atom * epsilon * 1e-15;  
    return dEdx_eV_per_cm * eV_to_J * 100.0;          
}

double energy_loss(mt_state* rng, double* final_energy, double dx, double n_atom, int Z, int z) {
    double n_atom_m3 = n_atom * 1e6;
    double dx_m = dx * 0.01;

    double coulomb_const = e_charge * e_charge / (4.0 * pi * epsilon_0);
    double sigma_sq = 4.0 * pi * z * z * coulomb_const * coulomb_const * Z * n_atom_m3 * dx_m;
    double sigma = sqrt(fmax(0.0, sigma_sq));

    double r1 = ranf_thread(rng), r2 = ranf_thread(rng);
    double gaussian_noise = sqrt(-2.0 * log(fmax(r1, 1e-16))) * cos(2.0 * pi * r2);

    double E_MeV = *final_energy / eV_to_J / 1e6;   
    double dEdx = dEdx_J_per_m(E_MeV);            
    double energy_loss_J = dEdx * dx_m + gaussian_noise * sigma;

    if (energy_loss_J < 0.0) energy_loss_J = 0.0;
    *final_energy -= energy_loss_J;
    if (*final_energy < 0.0) *final_energy = 0.0;

    return *final_energy;
}

void new_velocity(double* final_energy, double* v, double mass_incident) {
    double total_energy_J = *final_energy + mass_incident * speed_light * speed_light;
    if (*v < 0.01 * speed_light) {
        *v = sqrt(2.0 * *final_energy / mass_incident);
    } else {
        double gamma = total_energy_J / (mass_incident * speed_light * speed_light);
        *v = speed_light * sqrt(1.0 - 1.0 / (gamma * gamma));
    }
}

// Scattering and rotation
int scattering_determine(mt_state* rng, double b, double n_atom, double dx) {
    double cross_section = pi * b * b;           
    double prob = cross_section * n_atom * dx;   
    return (ranf_thread(rng) < prob) ? 1 : 0;
}

void rotate_direction(double d[3], double theta, double phi, double new_d[3]) {
    double u[3], perp[3];
    double wx = d[0], wy = d[1], wz = d[2];
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

    perp[0] = wy * u[2] - wz * u[1];
    perp[1] = wz * u[0] - wx * u[2];
    perp[2] = wx * u[1] - wy * u[0];
    double norm_perp = sqrt(perp[0]*perp[0] + perp[1]*perp[1] + perp[2]*perp[2]);
    if (norm_perp > 1e-12) { perp[0] /= norm_perp; perp[1] /= norm_perp; perp[2] /= norm_perp; }

    double st = sin(theta), ct = cos(theta);
    double sp = sin(phi), cp = cos(phi);

    new_d[0] = st * cp * u[0] + st * sp * perp[0] + ct * wx;
    new_d[1] = st * cp * u[1] + st * sp * perp[1] + ct * wy;
    new_d[2] = st * cp * u[2] + st * sp * perp[2] + ct * wz;

    double norm = sqrt(new_d[0]*new_d[0] + new_d[1]*new_d[1] + new_d[2]*new_d[2]);
    if (norm > 1e-12) { new_d[0] /= norm; new_d[1] /= norm; new_d[2] /= norm; }
}

void yes_new_position_and_direction(double* pos, double* dir, double dx, double n_atom,
                                    double* total_areal_density, double b, double E_scatter_J,
                                    double xi, double theta, double phi) {
    for (int j = 0; j < 3; j++) pos[j] += xi * dx * dir[j];
    *total_areal_density += n_atom * (xi * dx);

    double new_dir[3];
    rotate_direction(dir, theta, phi, new_dir);

    for (int j = 0; j < 3; j++) pos[j] += (1.0 - xi) * dx * new_dir[j];
    *total_areal_density += n_atom * ((1.0 - xi) * dx);

    for (int j = 0; j < 3; j++) dir[j] = new_dir[j];
}

void no_new_position_and_direction(double* pos, double* dir, double dx, double n_atom, double* total_areal_density) {
    for (int j = 0; j < 3; j++) pos[j] += dx * dir[j];
    *total_areal_density += n_atom * dx;
}


// Main simulation for one particle
void simulate_one_particle(mt_state* rng, double initial_energy, double initial_v,
                           double thickness, int N_layers, double dx, double n_atom,
                           int Z, int z, double mass_incident,
                           double* local_sum_weight, double* local_sum_energy_w,
                           double* local_sum_energy2_w, double* local_sum_angle_w,
                           double* local_sum_angle2_w, double* local_backscatter_weight,
                           double* local_hist, double* local_energy_hist_170,
                           double E_max_bins, double bin_width_hist) {
    double final_energy = initial_energy;
    double v_curr = initial_v;
    double pos[3] = {0.0, 0.0, 0.0};
    double dir[3] = {1.0, 0.0, 0.0};
    double current_weight = 1.0;
    double total_areal_density = 0.0;
    double b_max = 1e-10;   // cm

    for (int i = 0; i < N_layers; i++) {
        if (scattering_determine(rng, b_max, n_atom, dx)) {
            double u = ranf_thread(rng);
            double p = 2.0;   
            double b_actual = b_max * pow(u, p);   
            double weight = 4.0 * u * u * u;
            if (weight < 1e-30) weight = 1e-30;
            current_weight *= weight;  

            double xi = ranf_thread(rng);   

            double dx_m = dx * 0.01;
            double E_MeV_before = final_energy / eV_to_J / 1e6;
            double dEdx = dEdx_J_per_m(E_MeV_before);
            double energy_loss_to_scatter = dEdx * (xi * dx_m);
            double E_scatter_J = final_energy - energy_loss_to_scatter;
            if (E_scatter_J < 0) E_scatter_J = 0;

            double b_m = b_actual * 0.01;                  
            double theta = 2.0 * atan( (double)z * Z * e_charge * e_charge /
                                       (8.0 * pi * epsilon_0 * E_scatter_J * b_m) );
            double phi = 2.0 * pi * ranf_thread(rng);

            yes_new_position_and_direction(pos, dir, dx, n_atom, &total_areal_density,
                                           b_actual, E_scatter_J, xi, theta, phi);
        } else {
            no_new_position_and_direction(pos, dir, dx, n_atom, &total_areal_density);
        }
        
        final_energy = energy_loss(rng, &final_energy, dx, n_atom, Z, z);
        new_velocity(&final_energy, &v_curr, mass_incident);
        
        if (v_curr <= 0) break;
    }

    double E_MeV = final_energy / eV_to_J / 1e6;
    double dot = dir[0];
    if (dot > 1.0) dot = 1.0;
    if (dot < -1.0) dot = -1.0;
    double angle_deg = acos(dot) * 180.0 / pi;

    if (angle_deg >= 165.0 && angle_deg <= 175.0) {
        int e_bin = (int)(E_MeV / (E_max_bins / ENERGY_BINS_170));
        if (e_bin >= ENERGY_BINS_170) e_bin = ENERGY_BINS_170 - 1;
        if (e_bin >= 0) local_energy_hist_170[e_bin] += current_weight;
    }

    *local_sum_weight += current_weight;
    *local_sum_energy_w += current_weight * E_MeV;
    *local_sum_energy2_w += current_weight * E_MeV * E_MeV;
    *local_sum_angle_w += current_weight * angle_deg;
    *local_sum_angle2_w += current_weight * angle_deg * angle_deg;
    if (dir[0] < 0) *local_backscatter_weight += current_weight;

    int bin = (int)(angle_deg / bin_width_hist);
    if (bin >= HIST_BINS) bin = HIST_BINS - 1;
    if (bin >= 0) local_hist[bin] += current_weight;
}


// Input
void input_parameters() {
    int choice;
    printf("Select input mode:\n1. Load from csv files\n2. Manual input\nChoice: ");
    scanf("%d", &choice);

    if (choice == 1) {
        FILE *file = fopen("materials.csv", "r");
        if (!file) { printf("Error: Could not open materials.csv\n"); exit(1); }
        char buffer[256];
        fgets(buffer, sizeof(buffer), file);
        printf("\nAvailable materials:\n");
        while (fgets(buffer, sizeof(buffer), file)) {
            int num; char name[50];
            double Zmat, denval, Aval;
            double a0, a1, a2, a3, a4;
            if (sscanf(buffer, "%d, %49[^,], %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf",
                       &num, name, &Zmat, &denval, &Aval, &a0, &a1, &a2, &a3, &a4) == 10) {
                printf("%d: %s (Z=%.0f, ρ=%.2f g/cm^3, A=%.2f g/mol)\n", num, name, Zmat, denval, Aval);
            }
        }
        fclose(file);
        
        int mat_choice;
        printf("\nEnter material number: ");
        scanf("%d", &mat_choice);

        file = fopen("materials.csv", "r");
        fgets(buffer, sizeof(buffer), file);
        int found = 0;
        while (fgets(buffer, sizeof(buffer), file)) {
            int num; char name[50]; double Zmat, denval, Aval; double a0, a1, a2, a3, a4;
            if (sscanf(buffer, "%d, %49[^,], %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf",
                       &num, name, &Zmat, &denval, &Aval, &a0, &a1, &a2, &a3, &a4) == 10) {
                if (num == mat_choice) {
                    Z = (int)Zmat; den = denval; A = Aval;
                    eps_coeff[0] = a0; eps_coeff[1] = a1; eps_coeff[2] = a2;
                    eps_coeff[3] = a3; eps_coeff[4] = a4;
                    printf("Selected: %s\n", name); found = 1; break;
                }
            }
        }
        fclose(file);

        file = fopen("particles.csv", "r");
        if (!file) { printf("Error: Could not open particles.csv\n"); exit(1); }
        fgets(buffer, sizeof(buffer), file);
        printf("\nAvailable particles:\n");
        while (fgets(buffer, sizeof(buffer), file)) {
            int num; char name[50]; int zval; double vval, mval;
            if (sscanf(buffer, "%d, %49[^,], %d, %lf, %lf", &num, name, &zval, &vval, &mval) == 5) {
                printf("%d: %s (z=%d, v=%.2e m/s)\n", num, name, zval, vval);
            }
        }
        fclose(file);
        
        int part_choice;
        printf("\nEnter particle number: ");
        scanf("%d", &part_choice);

        file = fopen("particles.csv", "r");
        fgets(buffer, sizeof(buffer), file);
        while (fgets(buffer, sizeof(buffer), file)) {
            int num; char name[50]; int zval; double vval, mval;
            if (sscanf(buffer, "%d, %49[^,], %d, %lf, %lf", &num, name, &zval, &vval, &mval) == 5) {
                if (num == part_choice) {
                    z = zval; v = vval; mass_incident = mval;
                    printf("Selected: %s\n", name); break;
                }
            }
        }
        fclose(file);

        printf("\nEnter thickness (cm): "); scanf("%lf", &thickness);
        printf("Enter number of layers: "); scanf("%d", &N_layers);
    } else {
        printf("Enter atomic number (Z): "); scanf("%d", &Z);
        printf("Enter density (g/cm^3): "); scanf("%lf", &den);
        printf("Enter atomic mass (g/mol): "); scanf("%lf", &A);
        printf("Enter thickness (cm): "); scanf("%lf", &thickness);
        printf("Enter layers: "); scanf("%d", &N_layers);
        printf("Enter z: "); scanf("%d", &z);
        printf("Enter eps_coeff (a0 a1 a2 a3 a4):\n");
        scanf("%lf %lf %lf %lf %lf", &eps_coeff[0], &eps_coeff[1], &eps_coeff[2], &eps_coeff[3], &eps_coeff[4]);
        printf("Enter velocity (m/s): "); scanf("%lf", &v);
        printf("Enter rest mass (MeV/c^2): "); scanf("%lf", &mass_incident);
    }
    mass_incident = mass_incident * 1.78266192e-30; 
    initial_v = v;
    n_atom = N_A * den / A;
    dx = thickness / N_layers;
}

// Output 
void detector_output_histogram(const char *hist_filename, long long N_particles) {
    FILE *f = fopen(hist_filename, "w");
    if (!f) { printf("Error: Could not open %s\n", hist_filename); return; }
    fprintf(f, "# Histogram of scattering angles (degrees) – Weighted Importance Sampling (OpenMP)\n");
    fprintf(f, "# bins=%d, range=[0,180], bin_width=%.2f\n", HIST_BINS, bin_width);
    fprintf(f, "# Total simulated particles: %lld\n", N_particles);
    fprintf(f, "# Total weight (effective particles): %.3e\n", sum_weight);
    double total = 0.0;
    for (int i = 0; i < HIST_BINS; i++) total += hist[i];
    if (total > 0.0) {
        for (int i = 0; i < HIST_BINS; i++) {
            double prob_density = hist[i] / sum_weight / bin_width;
            double angle_center = (i + 0.5) * bin_width;
            fprintf(f, "%.4f %.6e\n", angle_center, prob_density);
        }
    }
    fclose(f);
}

void detector_output_energy_170(const char *filename, long long N_particles) {
    FILE *f = fopen(filename, "w");
    if (!f) return;
    double bin_width_e = E_max_bins / ENERGY_BINS_170;
    double total_weight_170 = 0.0;
    for (int i = 0; i < ENERGY_BINS_170; i++) total_weight_170 += energy_hist_170[i];
    fprintf(f, "# Energy spectrum for backscattering angle 170°±5° (OpenMP)\n");
    fprintf(f, "# bins=%d, range=[0,%.2f] MeV, bin_width=%.4f MeV\n", ENERGY_BINS_170, E_max_bins, bin_width_e);
    fprintf(f, "# Total weighted counts in window = %.3e\n", total_weight_170);
    fprintf(f, "# Simulated particles = %lld\n", N_particles);
    fprintf(f, "# Energy (MeV)   Weighted Counts\n");
    for (int i = 0; i < ENERGY_BINS_170; i++) {
        double E_center = (i + 0.5) * bin_width_e;
        fprintf(f, "%.4f %.6e\n", E_center, energy_hist_170[i]);
    }
    fclose(f);
}

void detector_output_results(const char *filename, long long N_particles) {
    FILE *outfile = fopen(filename, "w");
    if (!outfile) { printf("Error: Could not open output file %s\n", filename); return; }

    double mean_E = (sum_weight > 0) ? sum_energy_w / sum_weight : 0.0;
    double var_E = (sum_weight > 0) ? (sum_energy2_w / sum_weight - mean_E * mean_E) : 0.0;
    double mean_angle = (sum_weight > 0) ? sum_angle_w / sum_weight : 0.0;
    double var_angle = (sum_weight > 0) ? (sum_angle2_w / sum_weight - mean_angle * mean_angle) : 0.0;
    double back_prob = (sum_weight > 0) ? backscatter_weight / sum_weight : 0.0;

    fprintf(outfile, "Rutherford Scattering Simulation Results (Stable Importance Sampling - OpenMP)\n");
    fprintf(outfile, "===================================================================\n\n");
    fprintf(outfile, "Input Parameters:\n");
    fprintf(outfile, "-----------------\n");
    fprintf(outfile, "Material:\n");
    fprintf(outfile, "  Atomic number Z = %d\n", Z);
    fprintf(outfile, "  Density = %.2f g/cm³\n", den);
    fprintf(outfile, "  Atomic mass A = %.2f g/mol\n", A);
    fprintf(outfile, "  Atom density = %.2e atoms/cm³\n", n_atom);
    fprintf(outfile, "  Stopping cross section polynomial coefficients (1e-15 eV·cm², E in MeV):\n");
    fprintf(outfile, "    a0=%.4e, a1=%.4e, a2=%.4e, a3=%.4e, a4=%.4e\n",
        eps_coeff[0], eps_coeff[1], eps_coeff[2], eps_coeff[3], eps_coeff[4]);
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
    fprintf(outfile, "Simulation Results (Weighted):\n");
    fprintf(outfile, "----------------------------------------------\n");
    fprintf(outfile, "Total number of simulated particles: %lld\n", N_particles);
    fprintf(outfile, "Total weight (effective physical particles): %.3e\n", sum_weight);
    fprintf(outfile, "Final Energy (MeV):\n");
    fprintf(outfile, "  Mean = %.4f\n", mean_E);
    fprintf(outfile, "  Variance = %.4f\n", var_E);
    fprintf(outfile, "  Standard deviation = %.4f\n", sqrt(var_E));
    fprintf(outfile, "\nScattering Angle (degrees):\n");
    fprintf(outfile, "  Mean = %.4f\n", mean_angle);
    fprintf(outfile, "  Variance = %.4f\n", var_angle);
    fprintf(outfile, "  Standard deviation = %.4f\n", sqrt(var_angle));
    fprintf(outfile, "\nBackscattering (θ>90°):\n");
    fprintf(outfile, "  Weighted count = %.3e\n", backscatter_weight);
    fprintf(outfile, "  Probability = %.6e (1/%.0f)\n",
            back_prob, (back_prob > 0) ? 1.0/back_prob : 0.0);
    fclose(outfile);
    printf("Results saved to %s\n", filename);
}

// Main
int main() {
    input_parameters();

    initial_dir[0] = 1.0; initial_dir[1] = 0.0; initial_dir[2] = 0.0;

    if (initial_v < 0.01 * speed_light) {
        initial_energy = 0.5 * mass_incident * initial_v * initial_v;
    } else {
        double beta = initial_v / speed_light;
        double gamma = 1.0 / sqrt(1.0 - beta * beta);
        initial_energy = (gamma - 1.0) * mass_incident * speed_light * speed_light;
    }
    initial_energy_save = initial_energy;
    E_max_bins = initial_energy_save / 1.60218e-13;
    bin_width = 180.0 / HIST_BINS;

    long long N_particles;
    printf("Enter number of particles to simulate: ");
    scanf("%lld", &N_particles);
    printf("Simulating %lld particles\n", N_particles);

    uint32_t master_seed = 5489UL;
    printf("Enter random seed (default 5489): ");
    int tmp_seed;
    if (scanf("%d", &tmp_seed) == 1) master_seed = (uint32_t)tmp_seed;
    
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    printf("Using up to %d threads\n", num_threads);
#else
    int num_threads = 1;
    printf("OpenMP not enabled, using 1 thread\n");
#endif

    // Per‑thread private accumulators
    double *local_sum_weight = (double*)calloc(num_threads, sizeof(double));
    double *local_sum_energy_w = (double*)calloc(num_threads, sizeof(double));
    double *local_sum_energy2_w = (double*)calloc(num_threads, sizeof(double));
    double *local_sum_angle_w = (double*)calloc(num_threads, sizeof(double));
    double *local_sum_angle2_w = (double*)calloc(num_threads, sizeof(double));
    double *local_backscatter_weight = (double*)calloc(num_threads, sizeof(double));
    double **local_hist = (double**)malloc(num_threads * sizeof(double*));
    double **local_energy_hist_170 = (double**)malloc(num_threads * sizeof(double*));
    for (int t = 0; t < num_threads; t++) {
        local_hist[t] = (double*)calloc(HIST_BINS, sizeof(double));
        local_energy_hist_170[t] = (double*)calloc(ENERGY_BINS_170, sizeof(double));
    }

    // Progress counter and reporting interval
    volatile long long total_processed = 0;
    const long long PROGRESS_INTERVAL = 1000;   // update every 1000 particles

    #pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        mt_state rng;
        init_genrand_thread(&rng, master_seed + tid);
        
        double l_sum_w = 0.0, l_sum_e = 0.0, l_sum_e2 = 0.0;
        double l_sum_a = 0.0, l_sum_a2 = 0.0, l_back = 0.0;
        double* l_hist = local_hist[tid];
        double* l_ehist = local_energy_hist_170[tid];
        
        #pragma omp for schedule(dynamic, 1000)
        for (long long i = 0; i < N_particles; i++) {
            simulate_one_particle(&rng, initial_energy, initial_v,
                                  thickness, N_layers, dx, n_atom, Z, z, mass_incident,
                                  &l_sum_w, &l_sum_e, &l_sum_e2, &l_sum_a, &l_sum_a2, &l_back,
                                  l_hist, l_ehist, E_max_bins, bin_width);
            
            // Atomically increment global counter
            #pragma omp atomic
            total_processed++;
            
            // Only thread 0 prints progress at intervals
            if (tid == 0) {
                long long current = total_processed;
                
                if (current % PROGRESS_INTERVAL == 0 && current > 0) {
                    
                    static long long last_printed = 0;
                    if (current != last_printed) {
                        printf("\rProcessed %lld / %lld particles", current, N_particles);
                        fflush(stdout);
                        last_printed = current;
                    }
                }
            }
        }
        local_sum_weight[tid] = l_sum_w;
        local_sum_energy_w[tid] = l_sum_e;
        local_sum_energy2_w[tid] = l_sum_e2;
        local_sum_angle_w[tid] = l_sum_a;
        local_sum_angle2_w[tid] = l_sum_a2;
        local_backscatter_weight[tid] = l_back;
    }
    printf("\n"); 

    // Combine results
    sum_weight = 0.0;
    sum_energy_w = 0.0;
    sum_energy2_w = 0.0;
    sum_angle_w = 0.0;
    sum_angle2_w = 0.0;
    backscatter_weight = 0.0;
    for (int t = 0; t < num_threads; t++) {
        sum_weight += local_sum_weight[t];
        sum_energy_w += local_sum_energy_w[t];
        sum_energy2_w += local_sum_energy2_w[t];
        sum_angle_w += local_sum_angle_w[t];
        sum_angle2_w += local_sum_angle2_w[t];
        backscatter_weight += local_backscatter_weight[t];
        for (int i = 0; i < HIST_BINS; i++) hist[i] += local_hist[t][i];
        for (int i = 0; i < ENERGY_BINS_170; i++) energy_hist_170[i] += local_energy_hist_170[t][i];
    }

    // Free per‑thread arrays
    for (int t = 0; t < num_threads; t++) {
        free(local_hist[t]);
        free(local_energy_hist_170[t]);
    }
    free(local_hist);
    free(local_energy_hist_170);
    free(local_sum_weight);
    free(local_sum_energy_w);
    free(local_sum_energy2_w);
    free(local_sum_angle_w);
    free(local_sum_angle2_w);
    free(local_backscatter_weight);

    // Write outputs
    detector_output_histogram("histogram.csv", N_particles);
    detector_output_energy_170("energy_spectrum_170.csv", N_particles);
    detector_output_results("simulation_results.txt", N_particles);

    return 0;
}

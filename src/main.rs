use grid::MicromagneticSystem;
use std::f64;
mod grid;

/// Constants for the simulation
const DAMPING_CONSTANT: f64 = 0.5;
const GYROMAGNETIC_RATIO: f64 = 2.21e5;
const SATURATION_MAGNETIZATION: f64 = 8.0e5;
const EXCHANGE_STIFFNESS_CONSTANT: f64 = 1.3e-11;
const SPATIAL_DISCRETION_STEP: f64 = 1.0e-9;
const MAX_ITERATIONS_NUMBER: usize = 10000;
const TOLERANCE: f64 = 1e-6;
const UNIAXIAL_ANISOTROPY_CONSTANT: f64 = 1.0e4;
const EXTERNAL_FIELD: f64 = 0.1;
const EASY_AXIS: [f64; 1] = [1.0];

fn main() {
    // Number of cells in the 1D grid
    let number_of_cells = 100;

    // Initialize the micromagnetic system
    let mut system = MicromagneticSystem::new(number_of_cells);

    // Perform energy minimization
    system.minimize_energy();

    // Output the final magnetization state
    system.print_magnetization();
}

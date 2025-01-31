use magnetic_moments::MicromagneticSystem;
use export_to_excel::export;
use std::f64;
mod magnetic_moments;
mod export_to_excel;


/// Constants for the simulation
const DAMPING_CONSTANT: f64 = 0.5;
const GYROMAGNETIC_RATIO: f64 = 2.21e5;
const SATURATION_MAGNETIZATION: f64 = 8.0e5;
const EXCHANGE_STIFFNESS_CONSTANT: f64 = 1.3e-11;
const SPATIAL_DISCRETION_STEP: f64 = 1.0e-9;
const MAX_ITERATIONS_NUMBER: usize = 35000;
const TOLERANCE: f64 = 1e-6;
const UNIAXIAL_ANISOTROPY_CONSTANT: f64 = 1.0e4;
const EXTERNAL_FIELD: f64 = 0.5;
const EASY_AXIS: [f64; 3] = [0.0, 0.0, 1.0];

fn main() {
    // Number of cells in the 1D grid
    let number_of_cells = 50;

    // Initialize the micromagnetic system
    let mut system = MicromagneticSystem::new(number_of_cells);

    // Perform energy minimization
    system.minimize_energy();

    // Retrieve the normalized magnetization vectors
    let magnetizations = system.get_magnetizations();

    // Output the final magnetization state
    system.print_magnetizations();

    // Export the magnetization vectors to an Excel file
    if let Err(e) = export(magnetizations) {
        eprintln!("Failed to export magnetizations: {}", e);
    }
}
use magnetic_moments::MicromagneticSystem;
use export_to_excel::export;
use std::f64;
mod magnetic_moments;
mod export_to_excel;


/// Constants for the simulation

// Exchange interaction constants
const MAGNETIC_EXCHANGE_CONSTANT: f64 = 2.1e-11;
const SATURATION_MAGNETIZATION: f64 = 1.71e6;
const PERMEABILITY_OF_FREE_SPACE: f64 = 4.0 * f64::consts::PI * 1.0e-7;
const SPATIAL_DISCRETION_STEP: f64 = 1.0e-9;

// Anisotropy interaction constant 
const UNIAXIAL_ANISOTROPY_CONSTANT: f64 = 4.8e4;
const EASY_AXIS: [f64; 3] = [1.0, 0.0, 0.0];

// Zeeman interaction constant
const EXTERNAL_FIELD: [f64;3] = [0.0,0.0,0.5];

// Energy calculation constants
const TIME_STEP: f64 = 1e-15;
const DAMPING_CONSTANT: f64 = 0.2;
const GILBERT_GYROMAGNETIC_RATIO: f64 = 1.83e10;

// Iteration parameters
const MAX_ITERATIONS_NUMBER: usize = 10000;
const TOLERANCE: f64 = 1e-6;
  

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
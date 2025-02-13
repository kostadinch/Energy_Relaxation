use std::time;

use crate::DAMPING_CONSTANT;
use crate::EASY_AXIS;
use crate::EXTERNAL_FIELD;
use crate::GILBERT_GYROMAGNETIC_RATIO;
use crate::MAGNETIC_EXCHANGE_CONSTANT;
use crate::MAX_ITERATIONS_NUMBER;
use crate::PERMEABILITY_OF_FREE_SPACE;
use crate::SATURATION_MAGNETIZATION;
use crate::SPATIAL_DISCRETION_STEP;
use crate::TIME_STEP;
use crate::TOLERANCE;
use crate::UNIAXIAL_ANISOTROPY_CONSTANT;
use ndarray::{array, Array1};
use rand::Rng;

///# Micromagnetic System
/// Struct to represent the magnetic system
pub struct MicromagneticSystem {
    // Magnetization vectors
    magnetizations: Vec<Array1<f64>>,
    // Number particles
    size: usize,
}

impl MicromagneticSystem {
    ///# New Micromagnetic System
    /// Initialize the micromagnetic system with random magnetizations
    pub fn new(size: usize) -> Self {
        let mut magnetizations = vec![Array1::zeros(3); size];
        for i in 0..size {
            let mut rng = rand::rng();
            magnetizations[i][[0]] = rng.random_range(-1.0..=1.0);
            magnetizations[i][[1]] = rng.random_range(-1.0..=1.0);
            magnetizations[i][[2]] = rng.random_range(-1.0..=1.0);
            let norm = (magnetizations[i].dot(&magnetizations[i]) as f64).sqrt();
            magnetizations[i] /= norm;
        }
        // Create the system
        Self {
            magnetizations,
            size,
        }
    }

    ///# Total Effective Field Calculation
    /// Compute the total effective field at each cell by
    /// calculating and summing the exchange, anisotropy, and Zeeman fields.
    fn compute_effective_field(&self) -> Vec<Array1<f64>> {
        let mut h_eff: Vec<Array1<f64>> = vec![Array1::zeros(3); self.size];

        // Exchange Field Calculation
        // Finds the effective field at each cell using a finite difference method
        // for the gradient. The exchange field arises from the
        // quantum mechanical exchange interaction between neighboring spins,
        // which tends to align them to minimize energy.
        // This interaction smoothens spatial variations in magnetization and
        // penalizes sharp changes, creating a preference for uniform magnetization.
        for i in 1..(self.size - 1) {
            h_eff[i] = h_eff[i].clone()
                + (2.0 * MAGNETIC_EXCHANGE_CONSTANT
                    / (SATURATION_MAGNETIZATION * PERMEABILITY_OF_FREE_SPACE))
                    * (self.magnetizations[i + 1].clone() - 2.0 * self.magnetizations[i].clone()
                        + self.magnetizations[i - 1].clone())
                    / (SPATIAL_DISCRETION_STEP * SPATIAL_DISCRETION_STEP);
        }

        // Anisotropy Field Calculation
        // Calculates it based on a predetermined preferred direction of magnetization
        // (easy axis) and the magnetization at each cell.
        // The anisotropy field arises from the material's crystalline structure
        // or shape, which imposes a preferred direction (easy axis) for magnetization.
        // This preferred direction minimizes the anisotropy energy when the
        // magnetization aligns with it.
        for i in 0..self.size {
            //Dot product of the magnetization and the easy axis
            let scalar_product_of_the_magnetization_and_the_easy_axis =
                self.magnetizations[i].dot(&Array1::from_vec(EASY_AXIS.to_vec()));

            h_eff[i] = h_eff[i].clone()
                + 2.0
                    * UNIAXIAL_ANISOTROPY_CONSTANT
                    * scalar_product_of_the_magnetization_and_the_easy_axis
                    / (SATURATION_MAGNETIZATION * PERMEABILITY_OF_FREE_SPACE)
                    * Array1::from_vec(EASY_AXIS.to_vec());
        }

        // Zeeman Field
        // We take the Zeeman field as a constant external field in the z-direction.
        // The Zeeman field represents the interaction of the magnetization
        // with an external magnetic field. This interaction tries to
        // align the magnetization with the external field direction
        // to minimize the Zeeman energy.
        for i in 0..self.size {
            h_eff[i] = h_eff[i].clone()
                + Array1::from_vec(EXTERNAL_FIELD.to_vec()) / (PERMEABILITY_OF_FREE_SPACE);
        }

        // returns the total effective field
        h_eff
    }

    fn compute_magnetic_energy_density(&self) -> f64 {
        let mut magnetic_energy_density = 0.0;

        //Exchange energy
        for i in 1..(self.size - 1) {
            magnetic_energy_density += -MAGNETIC_EXCHANGE_CONSTANT
                * self.magnetizations[i].dot(&self.magnetizations[i + 1])
                / (SATURATION_MAGNETIZATION * PERMEABILITY_OF_FREE_SPACE);
        }

        //Anisotropy energy
        for i in 0..self.size {
            let scalar_product_of_the_magnetization_and_the_easy_axis =
                self.magnetizations[i].dot(&Array1::from_vec(EASY_AXIS.to_vec()));
            magnetic_energy_density += -UNIAXIAL_ANISOTROPY_CONSTANT
                * scalar_product_of_the_magnetization_and_the_easy_axis;
        }

        //Zeeman energy
        for i in 0..self.size {
            let external_field_dot_m =
                self.magnetizations[i].dot(&Array1::from_vec(EXTERNAL_FIELD.to_vec()));
            magnetic_energy_density += -external_field_dot_m;
        }

        magnetic_energy_density
    }

    fn compute_magnetization_change(
        &self,
    ) -> Vec<Array1<f64>> {
        let mut partial_derivative_of_the_magnetization_with_respect_to_time: Vec<Array1<f64>> =
            vec![Array1::zeros(3); self.size];
        let mut magnetization_change: Vec<Array1<f64>> = vec![Array1::zeros(3); self.size];

        let h_eff = self.compute_effective_field();
        for i in 0..self.size {
            let m = &self.magnetizations[i];
            let h = &h_eff[i];
            let m_cross_h = array![
                m[1] * h[2] - m[2] * h[1],
                m[2] * h[0] - m[0] * h[2],
                m[0] * h[1] - m[1] * h[0]
            ];
            let m_cross_m_cross_h = array![
                m[1] * m_cross_h[2] - m[2] * m_cross_h[1],
                m[2] * m_cross_h[0] - m[0] * m_cross_h[2],
                m[0] * m_cross_h[1] - m[1] * m_cross_h[0]
            ];
            partial_derivative_of_the_magnetization_with_respect_to_time[i] =
                -GILBERT_GYROMAGNETIC_RATIO / (1.0 + DAMPING_CONSTANT.powi(2))
                    * (m_cross_h + DAMPING_CONSTANT * m_cross_m_cross_h);
            magnetization_change[i] = TIME_STEP
                * &partial_derivative_of_the_magnetization_with_respect_to_time[i];
        }

        magnetization_change
    }

    fn compute_energy_change(&mut self) -> f64 {
        let magnetization_change = self.compute_magnetization_change();
        let h_eff = self.compute_effective_field();
        let mut energy_change = 0.0;
        for i in 0..self.size {
            let m = &self.magnetizations[i];
            let h = &h_eff[i];
            let h_dot_magnetization_change = h.dot(&magnetization_change[i]);
            energy_change=-h_dot_magnetization_change*SATURATION_MAGNETIZATION*PERMEABILITY_OF_FREE_SPACE;
        }
        energy_change
    }

    



    /// #Relaxation Step
    /// Perform a single relaxation step to minimize energy
    /// using the damping term of the Landau-Lifshitz-Gilbert equation
    /// and the computed effective field and check for convergence.
    /// Also, clamp the magnetization to [-1, 1] so that it is normalized.
    fn relaxation_step(&mut self) -> f64 {
        // calculate the effective field
        let h_eff = self.compute_effective_field();
        let mut max_change: f64 = 0.0;

        // Goes through each cell and updates the magnetization
        for i in 0..self.size {
            // Calculate the change in magnetization
            let change_of_magnetization = -DAMPING_CONSTANT
                * GILBERT_GYROMAGNETIC_RATIO
                * h_eff[i].clone()
                * SATURATION_MAGNETIZATION;

            // Calculate the maximum change in magnetization
            // and update the magnetization
            max_change = max_change.max(
                change_of_magnetization
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0, f64::max),
            );

            // Update magnetization and normalize it
            self.magnetizations[i] = &self.magnetizations[i] + &change_of_magnetization;
            let norm = self.magnetizations[i].dot(&self.magnetizations[i]).sqrt();
            self.magnetizations[i] /= norm;
        }

        max_change
    }

    ///# Energy Minimization check
    /// Checks if the energy has converged or if the maximum number
    /// of iterations has been reached.
    /// After the relaxation process, the energy function can be evaluated to
    /// confirm that the system has reached a minimal energy configuration.
    /// If energy stops decreasing between steps or falls below a tolerance,
    /// it’s a sign that the system has stabilized.
    pub fn minimize_energy(&mut self) {
        // Maximum number of iterations
        for iter in 0..MAX_ITERATIONS_NUMBER {
            let max_change = self.relaxation_step();
            if max_change < TOLERANCE {
                println!("Converged after {} iterations.", iter);
                return;
            }
        }
        println!(
            "Warning: Did not converge within {} iterations.",
            MAX_ITERATIONS_NUMBER
        );
    }

    ///# Print Magnetizations
    pub fn print_magnetizations(&self) {
        for (i, m) in self.magnetizations.iter().enumerate() {
            println!("Cell {}: m = {}", i, m);
        }
    }

    ///# Get Magnetizations
    pub fn get_magnetizations(&self) -> Vec<Array1<f64>> {
        self.magnetizations.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Test the initialization of the MicromagneticSystem
    fn test_initialization() {
        let size = 10;
        let system = MicromagneticSystem::new(size);
        assert_eq!(system.size, size);
        for m in &system.magnetizations {
            assert_eq!(m.len(), 3);
            assert!((m[0] - (2.0 * std::f64::consts::PI / size as f64).sin()).abs() < f64::EPSILON);
            assert!((m[1] - (2.0 * std::f64::consts::PI / size as f64).cos()).abs() < f64::EPSILON);
            assert!((m[2] - (std::f64::consts::PI / size as f64).sin()).abs() < f64::EPSILON);
        }
    }

    #[test]
    /// Test the effective field calculation
    fn test_effective_field() {
        let size = 10;
        let system = MicromagneticSystem::new(size);
        let h_eff = system.compute_effective_field();
        assert_eq!(h_eff.len(), size);
        // Check if the effective field is calculated correctly
        // This is a simple check, more detailed checks can be added
        for h in &h_eff {
            assert!(h.iter().all(|&x| x.abs() > 0.0));
        }
    }

    #[test]
    /// Test a single relaxation step
    fn test_relaxation_step() {
        let size = 10;
        let mut system = MicromagneticSystem::new(size);
        let max_change = system.relaxation_step();
        assert!(max_change > 0.0);
        // Check if the magnetization values are clamped between -1 and 1
        for m in &system.magnetizations {
            assert!(m.iter().all(|&x| x >= -1.0 && x <= 1.0));
        }
    }

    #[test]
    /// Test the energy minimization process
    fn test_minimize_energy() {
        let size = 10;
        let mut system = MicromagneticSystem::new(size);
        system.minimize_energy();
        // Check if the system has converged
        let max_change = system.relaxation_step();
        assert!(max_change < TOLERANCE);
    }

    #[test]
    /// Test the print_magnetizations function
    fn test_print_magnetizations() {
        let size = 10;
        let system = MicromagneticSystem::new(size);
        system.print_magnetizations();
        // This test just ensures that the function runs without panicking
    }

    #[test]
    /// Test the get_magnetizations function
    fn test_get_magnetizations() {
        let size = 10;
        let system = MicromagneticSystem::new(size);
        let magnetizations = system.get_magnetizations();
        assert_eq!(magnetizations.len(), size);
        for m in magnetizations {
            assert_eq!(m.len(), 3);
            assert!(m.iter().all(|&x| x >= -1.0 && x <= 1.0));
        }
    }
}

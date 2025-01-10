use crate::DAMPING_CONSTANT;
use crate::EASY_AXIS;
use crate::EXCHANGE_STIFFNESS_CONSTANT;
use crate::EXTERNAL_FIELD;
use crate::GYROMAGNETIC_RATIO;
use crate::MAX_ITERATIONS_NUMBER;
use crate::SATURATION_MAGNETIZATION;
use crate::SPATIAL_DISCRETION_STEP;
use crate::TOLERANCE;
use crate::UNIAXIAL_ANISOTROPY_CONSTANT;
use std::f64;

///# Micromagnetic System
/// Struct to represent the magnetic system
pub struct MicromagneticSystem {
    // Magnetization array (1D grid for simplicity)
    magnetization: Vec<f64>,
    // Number of cells in the grid
    size: usize,
}

impl MicromagneticSystem {
    ///# New Micromagnetic System
    /// Initialize the system with a uniform profile close to the easy axis
    pub fn new(size: usize) -> Self {
        let mut magnetization = Vec::with_capacity(size);
        // Initialize the magnetization array
        for _ in 0..size {
            // Start close to the easy axis with a slight perturbation
            let value = 0.95;
            magnetization.push(value);
        }
        // Create the system
        Self {
            magnetization,
            size,
        }
    }

    ///# Total Effective Field Calculation
    /// Compute the total effective field at each cell by
    /// calculating and summing the exchange, anisotropy, and Zeeman fields.
    fn compute_effective_field(&self) -> Vec<f64> {
        let mut h_eff = vec![0.0; self.size];

        // Exchange Field Calculation
        // Finds the effective field at each cell using a finite difference method
        // for the gradient. The exchange field arises from the
        // quantum mechanical exchange interaction between neighboring spins,
        // which tends to align them to minimize energy.
        // This interaction smoothens spatial variations in magnetization and
        // penalizes sharp changes, creating a preference for uniform magnetization.
        for i in 1..(self.size - 1) {
            h_eff[i] += (2.0 * EXCHANGE_STIFFNESS_CONSTANT / SATURATION_MAGNETIZATION)
                * (self.magnetization[i + 1] - 2.0 * self.magnetization[i]
                    + self.magnetization[i - 1])
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
            let scalar_product_of_the_magnetization_and_the_easy_axis =
                self.magnetization[i] * EASY_AXIS[0]; // Dot product with easy axis
            h_eff[i] += 2.0
                * UNIAXIAL_ANISOTROPY_CONSTANT
                * scalar_product_of_the_magnetization_and_the_easy_axis
                / SATURATION_MAGNETIZATION;
        }

        // Zeeman Field
        // We take the Zeeman field as a constant external field in the z-direction.
        // The Zeeman field represents the interaction of the magnetization
        // with an external magnetic field. This interaction tries to
        // align the magnetization with the external field direction
        // to minimize the Zeeman energy.
        for i in 0..self.size {
            h_eff[i] += EXTERNAL_FIELD;
        }

        // returns the total effective field
        h_eff
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
            let change_of_magnetization =
                -DAMPING_CONSTANT * GYROMAGNETIC_RATIO * h_eff[i] * SATURATION_MAGNETIZATION;
            max_change = max_change.max(change_of_magnetization.abs());

            // Update magnetization and clamp to [-1, 1] (normalized)
            self.magnetization[i] += change_of_magnetization;
            self.magnetization[i] = self.magnetization[i].clamp(-1.0, 1.0);
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

    /// Print the current magnetization state
    pub fn print_magnetization(&self) {
        for (i, &m) in self.magnetization.iter().enumerate() {
            println!("Cell {}: m = {:.6}", i, m);
        }
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
        for &m in &system.magnetization {
            assert!((m - 0.95).abs() < f64::EPSILON);
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
        for &h in &h_eff {
            assert!(h.abs() > 0.0);
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

        for &m in &system.magnetization {
            assert!(m >= -1.0 && m <= 1.0);
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
    /// Test the print_magnetization function
    fn test_print_magnetization() {
        let size = 10;
        let system = MicromagneticSystem::new(size);
        system.print_magnetization();
        // This test just ensures that the function runs without panicking
    }
}

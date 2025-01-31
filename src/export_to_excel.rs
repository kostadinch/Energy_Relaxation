use rust_xlsxwriter::Workbook;
use ndarray::Array1;
use std::error::Error;
use std::path::Path;

/// Export the magnetization vectors to an Excel file.
pub fn export(magnetizations: Vec<Array1<f64>>) -> Result<(), Box<dyn Error>> {

    // Create a new workbook and worksheet
    let path = Path::new("vectors.xlsx");
    let mut workbook = Workbook::new();
    let worksheet = workbook.add_worksheet();

    // Write header
    worksheet.write_row(0, 0, ["X", "Y", "Z"])?;

    // Write vector data
    // The first row is the header, so we start from the second row
    for (i, vector) in magnetizations.iter().enumerate() {
        worksheet.write_row(
            (i + 1) as u32,
            0,
            [
                vector[0] as f64,
                vector[1] as f64,
                vector[2] as f64,
            ]
        )?;
    }
    // Save the workbook
    workbook.save(path)?;

    Ok(())
}

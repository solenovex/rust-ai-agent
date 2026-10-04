use calamine::{Data, Reader, open_workbook_auto};

pub fn read_spreadsheet(file_path: &str) -> anyhow::Result<String> {
    let mut workbook = open_workbook_auto(file_path)?;

    let sheet_names = workbook.sheet_names().to_owned();

    let mut result = String::new();

    for sheet_name in sheet_names {
        result.push_str(&format!("## Sheet: {sheet_name}\n\n"));

        if let Ok(range) = workbook.worksheet_range(&sheet_name) {
            for row in range.rows() {
                let values: Vec<String> = row
                    .iter()
                    .map(|cell| match cell {
                        Data::Empty => String::new(),
                        Data::String(s) => s.clone(),
                        Data::Float(n) => n.to_string(),
                        Data::Int(n) => n.to_string(),
                        Data::Bool(b) => b.to_string(),
                        Data::Error(e) => format!("[ERROR: {e:?}]"),
                        Data::DateTime(dt) => dt.to_string(),
                        Data::DateTimeIso(s) => s.clone(),
                        Data::DurationIso(s) => s.clone(),
                    })
                    .collect();

                result.push_str("| ");
                result.push_str(&values.join(" | "));
                result.push_str(" |\n");
            }
        }

        result.push('\n');
    }

    Ok(result)
}

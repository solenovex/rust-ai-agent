use std::{fs, path::Path};

use crate::tools::file_read::r#impl::ReadFileArgs;

pub fn read(args: &ReadFileArgs) -> anyhow::Result<String> {
    let path = Path::new(&args.file_path);
    if !path.exists() {
        anyhow::bail!("File not found: {}", args.file_path);
    }

    match path.extension().and_then(|ext| ext.to_str()) {
        Some("csv") => read_csv_as_markdown(path),
        _ => read_text_with_line_numbers(path, args.start_line, args.end_line),
    }
}

fn read_text_with_line_numbers(path: &Path, start_line: usize, end_line: i64) -> anyhow::Result<String> {
    let content = fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();

    let start_index = start_line.saturating_sub(1);
    let end_index = if end_line < 0 {
        lines.len()
    } else {
        (end_line as usize) .min(lines.len())
    };

    let mut result = String::new();
    for (offset, line) in lines.get(start_index..end_index).unwrap_or(&[]).iter().enumerate() {
        result.push_str(&format!("{:>4} | {}\n", start_index + offset + 1, line));
    }

    Ok(result)
}

fn read_csv_as_markdown(path: &Path) -> anyhow::Result<String> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();

    let mut table = String::new();
    table.push_str(&format!("| {} |\n", headers.iter().collect::<Vec<_>>().join(" | ")));
    table.push_str(&format!("|{}\n", "---|".repeat(headers.len())));

    for record in reader.records() {
        let record = record?;
        table.push_str(&format!("| {} |\n", record.iter().collect::<Vec<_>>().join(" | ")));
    }

    Ok(table)
}

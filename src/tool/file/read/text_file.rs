use std::fs;

pub fn read_text_file(
    file_path: &str,
    start_line: usize,
    end_line: Option<usize>,
) -> anyhow::Result<String> {
    let content = fs::read_to_string(file_path)?;
    let lines: Vec<&str> = content.lines().collect();
    let start = start_line.saturating_sub(1);
    let end = end_line.unwrap_or(lines.len()).min(lines.len());

    if start >= end {
        return Ok(String::new());
    }

    let mut result = Vec::new();
    for (i, line) in lines[start..end].iter().enumerate() {
        let line_number = start + i + 1;
        result.push(format!("{:4} | {}", line_number, line.trim_end()));
    }
    Ok(result.join("\n"))
}

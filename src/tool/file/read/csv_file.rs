pub fn read_csv(path: &str) -> anyhow::Result<String> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();

    let mut table = String::new();
    table.push_str(&format!(
        "| {} |\n",
        headers.iter().collect::<Vec<_>>().join(" | ")
    ));
    table.push_str(&format!("|{}\n", "---|".repeat(headers.len())));

    for record in reader.records() {
        let record = record?;
        table.push_str(&format!(
            "| {} |\n",
            record.iter().collect::<Vec<_>>().join(" | ")
        ));
    }

    Ok(table)
}

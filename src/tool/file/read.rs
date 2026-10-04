use std::path::Path;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::{
    agent::context::ExecutionContext,
    tool::{
        Tool,
        file::read::{
            csv_file::read_csv, spreadsheet::read_spreadsheet, text_file::read_text_file,
        },
    },
};

pub mod csv_file;
pub mod spreadsheet;
pub mod text_file;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadFileArgs {
    pub file_path: String,

    #[serde(default = "default_start")]
    pub start_line: usize,

    #[serde(default)]
    pub end_line: Option<usize>,
}

fn default_start() -> usize {
    1
}

pub struct ReadFileTool;

#[async_trait::async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a text file (returned with line numbers, optionally a range) or a CSV file (returned as a markdown table)."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ReadFileArgs))
            .expect("schema is always serializable")
    }

    fn requires_confirmation(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: ReadFileArgs = serde_json::from_str(args_json)?;
        read_file(&args.file_path, args.start_line, args.end_line)
    }
}

const TEXT_EXTENSIONS: &[&str] = &[
    "txt", "py", "js", "json", "md", "html", "css", "xml", "yaml", "yml", "log", "sh",
];
const SPREADSHEET_EXTENSIONS: &[&str] = &["xlsx", "xls", "csv"];

pub fn read_file(
    file_path: &str,
    start_line: usize,
    end_line: Option<usize>,
) -> anyhow::Result<String> {
    let path = Path::new(file_path);
    if !path.exists() {
        anyhow::bail!("Path does not exist: {}", path.display());
    }

    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();

    if TEXT_EXTENSIONS.contains(&ext.as_str()) {
        read_text_file(file_path, start_line, end_line)
    } else if ext == "csv" {
        read_csv(file_path)
    } else if SPREADSHEET_EXTENSIONS.contains(&ext.as_str()) {
        read_spreadsheet(file_path)
    } else {
        anyhow::bail!("Unsupported file type: {}", path.display());
    }
}

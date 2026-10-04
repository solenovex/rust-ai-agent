use std::{fs, path::Path};

use schemars::JsonSchema;
use serde::Deserialize;

use crate::{agent::context::ExecutionContext, tool::Tool};

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListFilesArgs {
    #[serde(default = "default_path")]
    pub path: String,
}

fn default_path() -> String {
    ".".to_string()
}

pub struct ListFileTool;

#[async_trait::async_trait]
impl Tool for ListFileTool {
    fn name(&self) -> &str {
        "list_files"
    }

    fn description(&self) -> &str {
        "List files and directories at a given path, directories listed first."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ListFilesArgs))
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
        let args: ListFilesArgs = serde_json::from_str(args_json)?;
        match list_files(&args.path) {
            Ok(listing) => Ok(listing),
            Err(err) => Ok(format!("Error: {err}")),
        }
    }
}

pub fn list_files(path: &str) -> anyhow::Result<String> {
    let path = Path::new(path);
    if !path.exists() {
        anyhow::bail!("Path does not exist: {}", path.display());
    }
    if !path.is_dir() {
        anyhow::bail!("Path is not a directory: {}", path.display());
    }
    let mut items = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let item_path = entry.path();

        if item_path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|name| name.starts_with("."))
        {
            continue;
        }

        if item_path.is_dir() {
            items.push(format!("{}/", item_path.display()));
        } else {
            items.push(item_path.display().to_string());
        }
    }

    items.sort();
    let mut result = format!("Directory: {}\n", path.display());
    for item in items {
        result.push_str(&format!("\t{}\n", item));
    }

    Ok(result)
}

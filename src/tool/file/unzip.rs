use std::{
    fs, io,
    path::{Path, PathBuf},
};

use schemars::JsonSchema;
use serde::Deserialize;

use crate::{agent::context::ExecutionContext, tool::Tool};

#[derive(Deserialize, Debug, JsonSchema)]
pub struct UnzipFileArgs {
    pub zip_path: String,

    #[serde(default)]
    pub extract_to: Option<String>,
}

pub struct UnzipFileTool;

#[async_trait::async_trait]
impl Tool for UnzipFileTool {
    fn name(&self) -> &str {
        "unzip_file"
    }

    fn description(&self) -> &str {
        "Extract a zip archive so its contents can be explored with list_files and read_file."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(UnzipFileArgs))
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
        let args: UnzipFileArgs = serde_json::from_str(args_json)?;
        match unzip(&args.zip_path, args.extract_to.as_deref()) {
            Ok(summary) => Ok(summary),
            Err(err) => Ok(format!("Error: {err}")),
        }
    }
}

pub fn unzip(zip_file_path: &str, extract_to: Option<&str>) -> anyhow::Result<String> {
    let zip_file_path = Path::new(zip_file_path);
    if !zip_file_path.exists() {
        anyhow::bail!("File not found: {}", zip_file_path.display());
    }
    let extract_to: PathBuf = match extract_to {
        Some(dir) => PathBuf::from(dir),
        None => zip_file_path.with_extension(""),
    };

    fs::create_dir_all(&extract_to)?;

    let file = fs::File::open(zip_file_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut names = Vec::with_capacity(archive.len());
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let out_path = &extract_to.join(entry.name());
        names.push(entry.name().to_string());
        if entry.is_dir() {
            fs::create_dir_all(out_path)?;
        } else {
            let mut out_file = fs::File::create(out_path)?;
            io::copy(&mut entry, &mut out_file)?;
        }
    }

    let mut summary = format!(
        "Extracted {} files to {}/\n\nContents:\n",
        names.len(),
        extract_to.display()
    );
    for name in names.iter().take(20) {
        summary.push_str(&format!("{}\n", name));
    }
    if names.len() > 20 {
        summary.push_str(&format!("  ... and {} more files\n", names.len() - 20));
    }

    Ok(summary)
}

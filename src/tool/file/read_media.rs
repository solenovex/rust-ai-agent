use std::path::Path;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::{
    agent::context::ExecutionContext,
    constant,
    tool::{
        Tool,
        file::read_media::{audio::analyze_audio, image::analyze_image, pdf::analyze_pdf},
    },
};

pub mod audio;
pub mod image;
pub mod pdf;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadMediaArgs {
    pub file_path: String,
    pub query: String,
}

pub struct ReadMediaTool;

#[async_trait::async_trait]
impl Tool for ReadMediaTool {
    fn name(&self) -> &str {
        "read_media"
    }

    fn description(&self) -> &str {
        "Analyze an image, audio, or PDF file with an appropriate AI model to answer a question about its content."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ReadMediaArgs))
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
        let args: ReadMediaArgs = serde_json::from_str(args_json)?;
        read_media_file(&args.file_path, &args.query).await
    }
}

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp"];
const AUDIO_EXTENSIONS: &[&str] = &["mp3", "wav", "m4a", "flac", "ogg", "webm"];
const PDF_EXTENSIONS: &[&str] = &["pdf"];

pub async fn read_media_file(file_path: &str, query: &str) -> anyhow::Result<String> {
    let model = constant::GPT_6_LUNA_PRO_MODEL;
    let path = Path::new(file_path);
    if !path.exists() {
        anyhow::bail!("Path does not exist: {}", path.display());
    }

    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();

    if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
        analyze_image(file_path, query, model).await
    } else if AUDIO_EXTENSIONS.contains(&ext.as_str()) {
        analyze_audio(file_path, query, model).await
    } else if PDF_EXTENSIONS.contains(&ext.as_str()) {
        analyze_pdf(file_path, query, model).await
    } else {
        anyhow::bail!("Unsupported media format: {}", path.display());
    }
}

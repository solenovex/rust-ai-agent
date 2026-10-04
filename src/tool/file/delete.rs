use schemars::JsonSchema;
use serde::Deserialize;

use crate::{agent::context::ExecutionContext, tool::Tool};

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteFileArgs {
    pub file_path: String,
}

pub struct DeleteFileTool;

#[async_trait::async_trait]
impl Tool for DeleteFileTool {
    fn name(&self) -> &str {
        "delete_file"
    }

    fn description(&self) -> &str {
        "Deletes a file. This action cannot be undone."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(DeleteFileArgs))
            .expect("schema is always serializable")
    }

    fn requires_confirmation(&self) -> bool {
        true
    }

    fn get_confirmation_message(&self, arguments: &serde_json::Value) -> String {
        let file_path = serde_json::from_value::<DeleteFileArgs>(arguments.clone())
            .map(|r| r.file_path)
            .unwrap_or_else(|_| arguments.to_string());

        format!(
            "The agent wants to delete '{file_path}'. \
            This action cannot be undone. Do you approve?"
        )
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: DeleteFileArgs = serde_json::from_str(args_json)?;
        tracing::info!("🗑️  Attempting to delete: {}", args.file_path);
        match std::fs::remove_file(&args.file_path) {
            Ok(()) => {
                tracing::info!("✅ Deleted: {}", args.file_path);
                Ok(format!("File {} has been deleted.", args.file_path))
            }
            Err(e) => {
                tracing::error!("❌ Failed to delete {}: {e}", args.file_path);
                Err(anyhow::anyhow!("Failed to delete {}: {e}", args.file_path))
            }
        }
    }
}

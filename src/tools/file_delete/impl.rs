use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::{agent::ExecutionContext, tools::tool::Tool};

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

    fn parameters(&self) -> Value {
        serde_json::to_value(schemars::schema_for!(DeleteFileArgs))
            .expect("schema is always serializable")
    }

    fn requires_confirmation(&self) -> bool {
        true
    }

    fn confirmation_message_template(&self) -> &str {
        "即将执行不可逆的文件删除操作：{arguments}，是否批准？"
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

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::agent::context::ExecutionContext;
use crate::sandbox::{CodeExecutor, WorkspacePath};
use crate::tool::Tool;

/// Whitelist for host files the LLM may upload: only files under `allowed_root`
/// (after resolving symlinks) can be read.
#[derive(Debug, Clone)]
pub struct UploadPolicy {
    pub allowed_root: PathBuf,
}

impl UploadPolicy {
    /// Returns the canonical path if allowed, or a message for the LLM if not.
    pub fn check(&self, local: &Path) -> Result<PathBuf, String> {
        let root = self
            .allowed_root
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let real = local
            .canonicalize()
            .map_err(|_| format!("Local file not found: {}", local.display()))?;
        if !real.starts_with(&root) {
            return Err(format!(
                "Path is outside the allowed upload root: {}",
                local.display()
            ));
        }
        if !real.is_file() {
            return Err(format!("Path is not a file: {}", local.display()));
        }
        Ok(real)
    }
}

#[derive(serde::Deserialize)]
struct Args {
    local_path: String,
    #[serde(default)]
    sandbox_path: Option<String>,
}

/// Copies an allowed host file into the sandbox workspace.
pub struct UploadFileTool {
    executor: Arc<dyn CodeExecutor>,
    policy: UploadPolicy,
}

impl UploadFileTool {
    pub fn new(executor: Arc<dyn CodeExecutor>, policy: UploadPolicy) -> Self {
        Self { executor, policy }
    }
}

#[async_trait]
impl Tool for UploadFileTool {
    fn name(&self) -> &str {
        "upload_file"
    }

    fn description(&self) -> &str {
        "Upload a local file into the sandbox workspace (/workspace). Only files under the configured upload folder are allowed."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "local_path": { "type": "string", "description": "Absolute path of the local file" },
                "sandbox_path": { "type": "string", "description": "Destination path under /workspace (optional, defaults to the file name)" }
            },
            "required": ["local_path"]
        })
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: Args = serde_json::from_str(args_json)?;
        // Policy denials go back to the LLM as a tool result, not as a hard error.
        let real = match self.policy.check(Path::new(&args.local_path)) {
            Ok(p) => p,
            Err(msg) => return Ok(format!("Error: {msg}")),
        };
        let dest_raw = match &args.sandbox_path {
            Some(p) => p.clone(),
            None => real
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
        };
        let dest = WorkspacePath::parse(&dest_raw)?;
        let bytes = tokio::fs::read(&real).await?;
        self.executor.write_file(&dest, &bytes).await?;
        Ok(format!(
            "Successfully uploaded '{}' to '/workspace/{}'",
            args.local_path,
            dest.as_path().display()
        ))
    }

    fn requires_confirmation(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::context::ExecutionContext;
    use crate::tool::code::test_support::FakeExecutor;

    fn tool_for(root: &std::path::Path, fake: Arc<FakeExecutor>) -> UploadFileTool {
        UploadFileTool::new(
            fake,
            UploadPolicy {
                allowed_root: root.to_path_buf(),
            },
        )
    }

    #[test]
    fn policy_allows_files_under_the_root() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("data.xlsx"), b"123").unwrap();
        let policy = UploadPolicy {
            allowed_root: root.path().to_path_buf(),
        };
        assert!(policy.check(&root.path().join("data.xlsx")).is_ok());
    }

    #[test]
    fn policy_rejects_files_outside_the_root() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret"), b"x").unwrap();
        let policy = UploadPolicy {
            allowed_root: root.path().to_path_buf(),
        };
        let err = policy.check(&outside.path().join("secret")).unwrap_err();
        assert!(err.starts_with("Path is outside"));
    }

    #[cfg(unix)]
    #[test]
    fn policy_rejects_a_symlink_that_points_outside_the_root() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret"), b"x").unwrap();
        std::os::unix::fs::symlink(outside.path().join("secret"), root.path().join("link"))
            .unwrap();
        let policy = UploadPolicy {
            allowed_root: root.path().to_path_buf(),
        };
        assert!(policy.check(&root.path().join("link")).is_err());
    }

    #[tokio::test]
    async fn uploads_an_allowed_file_into_the_workspace_by_name() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("data.xlsx"), b"123").unwrap();
        let fake = Arc::new(FakeExecutor::new());
        let tool = tool_for(root.path(), fake.clone());
        let args = serde_json::json!({ "local_path": root.path().join("data.xlsx") }).to_string();
        let out = tool
            .execute(&args, &ExecutionContext::default())
            .await
            .unwrap();
        assert!(out.starts_with("Successfully uploaded"));
        let writes = fake.writes.lock().unwrap();
        assert_eq!(writes[0], ("data.xlsx".to_string(), b"123".to_vec()));
    }

    #[tokio::test]
    async fn denied_upload_is_returned_to_the_llm_not_raised() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret"), b"x").unwrap();
        let fake = Arc::new(FakeExecutor::new());
        let tool = tool_for(root.path(), fake.clone());
        let args = serde_json::json!({ "local_path": outside.path().join("secret") }).to_string();
        let out = tool
            .execute(&args, &ExecutionContext::default())
            .await
            .unwrap();
        assert!(out.starts_with("Error: Path is outside"));
        assert!(fake.writes.lock().unwrap().is_empty());
    }

    #[test]
    fn requires_confirmation_because_it_reads_host_files() {
        let root = tempfile::tempdir().unwrap();
        let tool = tool_for(root.path(), Arc::new(FakeExecutor::new()));
        assert!(tool.requires_confirmation());
        assert_eq!(tool.name(), "upload_file");
    }
}

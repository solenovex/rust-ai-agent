use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::agent::context::ExecutionContext;
use crate::sandbox::{CodeExecutor, CommandResult};
use crate::tool::Tool;

#[derive(serde::Deserialize)]
struct Args {
    command: String,
}

/// Renders stdout, stderr and exit code as labelled sections for the LLM.
pub fn format_command_result(r: &CommandResult) -> String {
    let mut parts = Vec::new();
    if !r.stdout.is_empty() {
        parts.push(format!("[stdout]\n{}", r.stdout));
    }
    if !r.stderr.is_empty() {
        parts.push(format!("[stderr]\n{}", r.stderr));
    }
    parts.push(format!("[exit_code] {}", r.exit_code));
    parts.join("\n\n")
}

/// Runs a bash command in the sandbox container.
pub struct BashTool {
    executor: Arc<dyn CodeExecutor>,
}

impl BashTool {
    pub fn new(executor: Arc<dyn CodeExecutor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash_tool"
    }

    fn description(&self) -> &str {
        "Execute a bash command in the sandboxed container. The workspace is at /workspace. Network access is usually disabled."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "command": { "type": "string", "description": "Bash command to run" } },
            "required": ["command"]
        })
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: Args = serde_json::from_str(args_json)?;
        let result = self.executor.run_bash(&args.command).await?;
        Ok(format_command_result(&result))
    }

    fn requires_confirmation(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::context::ExecutionContext;
    use crate::sandbox::CommandResult;
    use crate::tool::code::test_support::FakeExecutor;

    #[test]
    fn format_matches_the_book_layout() {
        let s = format_command_result(&CommandResult {
            stdout: "a\n".into(),
            stderr: "warn\n".into(),
            exit_code: 2,
        });
        assert_eq!(s, "[stdout]\na\n\n\n[stderr]\nwarn\n\n\n[exit_code] 2");
    }

    #[test]
    fn empty_streams_are_omitted() {
        let s = format_command_result(&CommandResult {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 0,
        });
        assert_eq!(s, "[exit_code] 0");
    }

    #[tokio::test]
    async fn execute_formats_the_executor_reply() {
        let mut fake = FakeExecutor::new();
        fake.command_reply = CommandResult {
            stdout: "hi\n".into(),
            stderr: String::new(),
            exit_code: 0,
        };
        let tool = BashTool::new(Arc::new(fake));
        let out = tool
            .execute(r#"{"command":"echo hi"}"#, &ExecutionContext::default())
            .await
            .unwrap();
        assert_eq!(out, "[stdout]\nhi\n\n\n[exit_code] 0");
    }

    #[test]
    fn requires_confirmation_because_it_runs_shell() {
        let tool = BashTool::new(Arc::new(FakeExecutor::new()));
        assert!(tool.requires_confirmation());
        assert_eq!(tool.name(), "bash_tool");
    }
}

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::agent::context::ExecutionContext;
use crate::sandbox::CodeExecutor;
use crate::tool::Tool;

#[derive(serde::Deserialize)]
struct Args {
    code: String,
}

/// Runs Python in the sandbox; state persists across calls.
pub struct ExecutePythonTool {
    executor: Arc<dyn CodeExecutor>,
}

impl ExecutePythonTool {
    pub fn new(executor: Arc<dyn CodeExecutor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for ExecutePythonTool {
    fn name(&self) -> &str {
        "execute_python"
    }

    fn description(&self) -> &str {
        "Execute Python code in a sandboxed container. Variables and functions persist between calls. Returns stdout, stderr, the value of the last expression, and any error."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "code": { "type": "string", "description": "Python source to run" } },
            "required": ["code"]
        })
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        let args: Args = serde_json::from_str(args_json)?;
        let result = self.executor.run_python(&args.code).await?;
        Ok(result.to_json())
    }

    fn requires_confirmation(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::context::ExecutionContext;
    use crate::sandbox::{ExecutionError, ExecutionResult};
    use crate::tool::code::test_support::FakeExecutor;

    #[tokio::test]
    async fn returns_json_the_llm_can_read_including_errors() {
        let mut fake = FakeExecutor::new();
        fake.python_reply = ExecutionResult {
            stdout: "ready\n".into(),
            error: Some(ExecutionError {
                name: "ZeroDivisionError".into(),
                value: "division by zero".into(),
                traceback: String::new(),
            }),
            ..Default::default()
        };
        let tool = ExecutePythonTool::new(Arc::new(fake));
        let out = tool
            .execute(r#"{"code":"x = 1/0"}"#, &ExecutionContext::default())
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error"]["name"], "ZeroDivisionError");
        assert_eq!(v["stdout"], "ready\n");
    }

    #[tokio::test]
    async fn bad_arguments_are_an_error_not_a_panic() {
        let tool = ExecutePythonTool::new(Arc::new(FakeExecutor::new()));
        assert!(
            tool.execute("not json", &ExecutionContext::default())
                .await
                .is_err()
        );
    }

    #[test]
    fn requires_confirmation_because_it_runs_code() {
        let tool = ExecutePythonTool::new(Arc::new(FakeExecutor::new()));
        assert!(tool.requires_confirmation());
        assert_eq!(tool.name(), "execute_python");
    }
}

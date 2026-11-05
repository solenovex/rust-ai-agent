pub mod bridge;
pub mod config;
pub mod podman;
pub mod runner;
pub mod workspace;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub use workspace::WorkspacePath;

#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("podman error: {0}")]
    Podman(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("execution exceeded the hard timeout; the sandbox session must be recreated")]
    HardTimeout,
    #[error("invalid workspace path: {0}")]
    InvalidPath(String),
}

/// Value object: an exception raised by the sandboxed Python code.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutionError {
    pub name: String,
    pub value: String,
    #[serde(default)]
    pub traceback: String,
}

/// Value object: result of one Python execution.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutionResult {
    #[serde(default)]
    pub stdout: String,
    #[serde(default)]
    pub stderr: String,
    /// repr of the last expression, if any
    #[serde(default)]
    pub value: Option<String>,
    #[serde(default)]
    pub error: Option<ExecutionError>,
}

impl ExecutionResult {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("ExecutionResult is always serializable")
    }
}

/// Value object: result of one bash command.
#[derive(Debug, Clone, PartialEq)]
pub struct CommandResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// Domain boundary: the agent depends only on this trait, not on Podman.
#[async_trait]
pub trait CodeExecutor: Send + Sync {
    async fn run_python(&self, code: &str) -> Result<ExecutionResult, SandboxError>;
    async fn run_bash(&self, command: &str) -> Result<CommandResult, SandboxError>;
    async fn write_file(&self, path: &WorkspacePath, bytes: &[u8]) -> Result<(), SandboxError>;
    async fn shutdown(&self) -> Result<(), SandboxError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_result_json_has_error_name_for_llm() {
        let r = ExecutionResult {
            stdout: "ready\n".into(),
            error: Some(ExecutionError {
                name: "ZeroDivisionError".into(),
                value: "division by zero".into(),
                traceback: String::new(),
            }),
            ..Default::default()
        };
        let v: serde_json::Value = serde_json::from_str(&r.to_json()).unwrap();
        assert_eq!(v["error"]["name"], "ZeroDivisionError");
        assert_eq!(v["stdout"], "ready\n");
    }
}

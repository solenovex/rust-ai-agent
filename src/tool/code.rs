pub mod bash;
pub mod execute_python;
pub mod upload;
pub mod wikipedia;

use std::sync::Arc;

use self::bash::BashTool;
use self::execute_python::ExecutePythonTool;
use self::upload::{UploadFileTool, UploadPolicy};
use self::wikipedia::WikipediaRevisions;
use crate::sandbox::CodeExecutor;
use crate::sandbox::bridge::{SandboxTool, sandbox_tools_prompt};
use crate::sandbox::config::{CodeExecutionSettings, settings_from};
use crate::sandbox::podman::{PodmanConfig, PodmanSandbox};
use crate::skill::{discover_skills, generate_skills_prompt};
use crate::tool::{Tool, ToolBox};

/// Host functions that sandboxed Python code may call.
pub fn sandbox_tool_set() -> Vec<Arc<dyn SandboxTool>> {
    vec![Arc::new(WikipediaRevisions::new())]
}

/// Starts the sandbox and registers `execute_python`, `bash_tool` and, if an upload root
/// is configured, `upload_file`. Returns the system-prompt text describing the sandbox
/// functions (empty when `CODE_EXECUTION` is off).
pub async fn register_code_tools(toolbox: &mut ToolBox) -> anyhow::Result<String> {
    let Some(settings) = settings_from(|k| std::env::var(k).ok()).map_err(anyhow::Error::msg)?
    else {
        return Ok(String::new());
    };
    let CodeExecutionSettings {
        workspace,
        upload_root,
        skills_dir,
    } = settings;
    let tools = sandbox_tool_set();
    let mut cfg = PodmanConfig::new(workspace);
    cfg.skills_dir = skills_dir.clone();
    let sandbox: Arc<dyn CodeExecutor> =
        Arc::new(PodmanSandbox::start_with_tools(cfg, tools.clone()).await?);

    let python = ExecutePythonTool::new(sandbox.clone());
    toolbox.insert(python.name().to_owned(), Box::new(python));

    let bash = BashTool::new(sandbox.clone());
    toolbox.insert(bash.name().to_owned(), Box::new(bash));

    if let Some(root) = upload_root {
        let upload = UploadFileTool::new(sandbox.clone(), UploadPolicy { allowed_root: root });
        toolbox.insert(upload.name().to_owned(), Box::new(upload));
    }
    let skills = skills_dir
        .as_deref()
        .map(discover_skills)
        .unwrap_or_default();
    let skills_prompt = generate_skills_prompt(&skills, "/skills");
    let mut prompt = sandbox_tools_prompt(&tools);
    if !skills_prompt.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(&skills_prompt);
    }
    Ok(prompt)
}

#[cfg(test)]
mod register_tests {
    use super::*;

    #[test]
    fn sandbox_tool_set_includes_wikipedia_revisions() {
        let names: Vec<String> = sandbox_tool_set()
            .iter()
            .map(|t| t.name().to_string())
            .collect();
        assert!(
            names.contains(&"get_wikipedia_revisions".to_string()),
            "{names:?}"
        );
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex;

    use async_trait::async_trait;

    use crate::sandbox::{
        CodeExecutor, CommandResult, ExecutionResult, SandboxError, WorkspacePath,
    };

    /// Test double: returns canned results and records file writes.
    pub struct FakeExecutor {
        pub python_reply: ExecutionResult,
        pub command_reply: CommandResult,
        pub writes: Mutex<Vec<(String, Vec<u8>)>>,
    }

    impl FakeExecutor {
        pub fn new() -> Self {
            Self {
                python_reply: ExecutionResult::default(),
                command_reply: CommandResult {
                    stdout: String::new(),
                    stderr: String::new(),
                    exit_code: 0,
                },
                writes: Mutex::new(vec![]),
            }
        }
    }

    #[async_trait]
    impl CodeExecutor for FakeExecutor {
        async fn run_python(&self, _code: &str) -> Result<ExecutionResult, SandboxError> {
            Ok(self.python_reply.clone())
        }
        async fn run_bash(&self, _command: &str) -> Result<CommandResult, SandboxError> {
            Ok(self.command_reply.clone())
        }
        async fn write_file(&self, path: &WorkspacePath, bytes: &[u8]) -> Result<(), SandboxError> {
            self.writes
                .lock()
                .unwrap()
                .push((path.as_path().display().to_string(), bytes.to_vec()));
            Ok(())
        }
        async fn shutdown(&self) -> Result<(), SandboxError> {
            Ok(())
        }
    }
}

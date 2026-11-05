use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::sandbox::bridge::{SandboxTool, dispatch, python_stub};
use crate::sandbox::runner::RUNNER_PY;
use crate::sandbox::{CodeExecutor, CommandResult, ExecutionResult, SandboxError, WorkspacePath};

#[derive(Debug, Clone, PartialEq)]
pub enum NetworkPolicy {
    Disabled,
    Enabled,
}

#[derive(Debug, Clone)]
pub struct PodmanConfig {
    /// "podman" (default); "docker" also works, minus the Podman-only flags
    pub binary: String,
    pub image: String,
    pub workspace: PathBuf,
    pub skills_dir: Option<PathBuf>,
    pub memory: String,
    pub cpus: f32,
    pub pids_limit: u32,
    pub network: NetworkPolicy,
    pub read_only_rootfs: bool,
    /// Lifetime of the whole session (container is killed after this)
    pub lifetime: Duration,
    /// Timeout for a single execution
    pub exec_timeout: Duration,
    /// Environment variables injected into the container
    pub env: Vec<(String, String)>,
}

impl PodmanConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            binary: "podman".into(),
            image: "localhost/ai-agent-sandbox:latest".into(),
            workspace: workspace.into(),
            skills_dir: None,
            memory: "512m".into(),
            cpus: 1.0,
            pids_limit: 256,
            network: NetworkPolicy::Disabled,
            read_only_rootfs: true,
            lifetime: Duration::from_secs(300),
            exec_timeout: Duration::from_secs(30),
            env: vec![],
        }
    }
}

/// Pure function: builds the `podman run` argument list (testable without a container).
pub fn build_run_args(cfg: &PodmanConfig, name: &str, runtime_dir: &Path) -> Vec<String> {
    let mut a: Vec<String> = vec![
        "run".into(),
        "-d".into(),
        "--rm".into(),
        format!("--name={name}"),
        format!("--memory={}", cfg.memory),
        format!("--cpus={}", cfg.cpus),
        format!("--pids-limit={}", cfg.pids_limit),
        "--cap-drop=ALL".into(),
        "--security-opt=no-new-privileges".into(),
        "-w".into(),
        "/workspace".into(),
        "-v".into(),
        format!("{}:/workspace", cfg.workspace.display()),
        "-v".into(),
        // runner is mounted read-only so LLM-generated code cannot modify it
        format!("{}:/opt/agent:ro", runtime_dir.display()),
    ];
    if cfg.network == NetworkPolicy::Disabled {
        a.push("--network=none".into());
    }
    if cfg.read_only_rootfs {
        a.push("--read-only".into());
        a.push("--tmpfs=/tmp:rw,size=64m".into());
    }
    if let Some(skills) = &cfg.skills_dir {
        a.push("-v".into());
        a.push(format!("{}:/skills:ro", skills.display()));
    }
    if cfg.binary.ends_with("podman") {
        a.push(format!("--timeout={}", cfg.lifetime.as_secs()));
    }
    for (k, v) in &cfg.env {
        a.push("-e".into());
        a.push(format!("{k}={v}"));
    }
    a.push(cfg.image.clone());
    a.push("sleep".into());
    a.push("infinity".into());
    a
}

/// Rejects any existing symlink along `rel` (relative to `base`).
/// The workspace is a bind mount, so sandboxed code can create symlinks
/// that point outside it; the host must not follow them when writing.
fn reject_symlinks(base: &Path, rel: &Path) -> Result<(), SandboxError> {
    let mut cur = base.to_path_buf();
    for c in rel.components() {
        cur.push(c);
        if let Ok(meta) = std::fs::symlink_metadata(&cur)
            && meta.file_type().is_symlink() {
                return Err(SandboxError::InvalidPath(format!(
                    "symlink in path: {}",
                    cur.display()
                )));
            }
    }
    Ok(())
}

/// Writes `bytes` to `root/path` on the host, refusing to follow symlinks.
pub fn write_into_workspace(
    root: &Path,
    path: &WorkspacePath,
    bytes: &[u8],
) -> Result<(), SandboxError> {
    reject_symlinks(root, path.as_path())?;
    let dest = root.join(path.as_path());
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(dest, bytes)?;
    Ok(())
}

/// Host side of the JSON-line protocol to the in-container runner.
struct Runner {
    #[allow(dead_code)]
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
    next_id: u64,
}

impl Runner {
    async fn send(&mut self, v: &Value) -> Result<(), SandboxError> {
        let mut line =
            serde_json::to_string(v).map_err(|e| SandboxError::Protocol(e.to_string()))?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn recv(&mut self) -> Result<Value, SandboxError> {
        let line = self
            .stdout
            .next_line()
            .await?
            .ok_or_else(|| SandboxError::Protocol("runner closed its stdout".into()))?;
        serde_json::from_str(&line).map_err(|e| SandboxError::Protocol(e.to_string()))
    }
}

/// A long-lived Podman container with a resident Python runner (state is kept
/// across `run_python` calls).
pub struct PodmanSandbox {
    cfg: PodmanConfig,
    name: String,
    _runtime_dir: tempfile::TempDir,
    tools: HashMap<String, Arc<dyn SandboxTool>>,
    runner: Mutex<Runner>,
}

async fn remove_container(binary: &str, name: &str) {
    let _ = Command::new(binary)
        .args(["rm", "-f", "-t", "0", name])
        .output()
        .await;
}

impl PodmanSandbox {
    pub async fn start(cfg: PodmanConfig) -> Result<Self, SandboxError> {
        Self::start_with_tools(cfg, Vec::new()).await
    }

    /// Starts the sandbox and pre-registers `tools` as Python functions.
    pub async fn start_with_tools(
        cfg: PodmanConfig,
        tools: Vec<Arc<dyn SandboxTool>>,
    ) -> Result<Self, SandboxError> {
        let name = format!("agent-sandbox-{}", uuid::Uuid::new_v4().simple());
        std::fs::create_dir_all(&cfg.workspace)?;
        let runtime_dir = tempfile::tempdir()?;
        std::fs::write(runtime_dir.path().join("runner.py"), RUNNER_PY)?;

        // 1) start the long-lived container (sleep infinity)
        let args = build_run_args(&cfg, &name, runtime_dir.path());
        let out = Command::new(&cfg.binary).args(&args).output().await?;
        if !out.status.success() {
            return Err(SandboxError::Podman(
                String::from_utf8_lossy(&out.stderr).into_owned(),
            ));
        }

        // 2) start the resident runner inside it, speaking JSON lines over stdin/stdout
        let spawned = Command::new(&cfg.binary)
            .args([
                "exec",
                "-i",
                name.as_str(),
                "python",
                "-u",
                "/opt/agent/runner.py",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn();
        let mut child = match spawned {
            Ok(c) => c,
            Err(e) => {
                remove_container(&cfg.binary, &name).await;
                return Err(e.into());
            }
        };
        let (stdin, stdout) = match (child.stdin.take(), child.stdout.take()) {
            (Some(i), Some(o)) => (i, o),
            _ => {
                remove_container(&cfg.binary, &name).await;
                return Err(SandboxError::Protocol("runner pipes unavailable".into()));
            }
        };

        let sandbox = Self {
            cfg,
            name,
            _runtime_dir: runtime_dir,
            tools: tools
                .iter()
                .map(|t| (t.name().to_string(), t.clone()))
                .collect(),
            runner: Mutex::new(Runner {
                child,
                stdin,
                stdout: BufReader::new(stdout).lines(),
                next_id: 0,
            }),
        };

        // Install one Python stub per tool. If this fails, `sandbox` is dropped and its container removed.
        for t in &tools {
            let r = sandbox.run_python(&python_stub(t.as_ref())).await?;
            if let Some(e) = r.error {
                return Err(SandboxError::Protocol(format!(
                    "failed to register sandbox tool {}: {}",
                    t.name(),
                    e.value
                )));
            }
        }
        Ok(sandbox)
    }

    /// Reads messages until the result for `id` arrives, serving `tool_call` requests on the way.
    async fn exchange(&self, r: &mut Runner, id: u64) -> Result<ExecutionResult, SandboxError> {
        loop {
            let msg = r.recv().await?;
            match msg["type"].as_str() {
                Some("tool_call") => {
                    let reply = dispatch(&self.tools, &msg).await;
                    r.send(&reply).await?;
                }
                Some("result") if msg["id"].as_u64() == Some(id) => {
                    return serde_json::from_value(msg)
                        .map_err(|e| SandboxError::Protocol(e.to_string()));
                }
                _ => {} // stale or unknown message: ignore
            }
        }
    }
}

#[async_trait]
impl CodeExecutor for PodmanSandbox {
    async fn run_python(&self, code: &str) -> Result<ExecutionResult, SandboxError> {
        let mut r = self.runner.lock().await;
        r.next_id += 1;
        let id = r.next_id;
        r.send(&json!({
            "type": "exec",
            "id": id,
            "code": code,
            "timeout": self.cfg.exec_timeout.as_secs(),
        }))
        .await?;
        // The in-container SIGALRM is the soft timeout; this host-side deadline is the backstop.
        let hard = self.cfg.exec_timeout + Duration::from_secs(5);
        match tokio::time::timeout(hard, self.exchange(&mut r, id)).await {
            Ok(res) => res,
            Err(_) => Err(SandboxError::HardTimeout),
        }
    }

    async fn run_bash(&self, command: &str) -> Result<CommandResult, SandboxError> {
        // `podman exec` has no kill-on-timeout, so coreutils `timeout` inside the container does it.
        let secs = self.cfg.exec_timeout.as_secs().to_string();
        let out = Command::new(&self.cfg.binary)
            .args([
                "exec",
                self.name.as_str(),
                "timeout",
                "-s",
                "KILL",
                secs.as_str(),
                "bash",
                "-lc",
                command,
            ])
            .output()
            .await?;
        Ok(CommandResult {
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            exit_code: out.status.code().unwrap_or(-1),
        })
    }

    async fn write_file(&self, path: &WorkspacePath, bytes: &[u8]) -> Result<(), SandboxError> {
        write_into_workspace(&self.cfg.workspace, path, bytes)
    }

    async fn shutdown(&self) -> Result<(), SandboxError> {
        remove_container(&self.cfg.binary, &self.name).await;
        Ok(())
    }
}

impl Drop for PodmanSandbox {
    /// Safety net for early returns, panics and cancellation. Drop cannot await,
    /// so this uses the blocking std::process API.
    fn drop(&mut self) {
        let _ = std::process::Command::new(&self.cfg.binary)
            .args(["rm", "-f", "-t", "0", self.name.as_str()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn has(args: &[String], s: &str) -> bool {
        args.iter().any(|a| a == s)
    }

    #[test]
    fn default_run_args_are_locked_down() {
        let cfg = PodmanConfig::new("/tmp/ws");
        let a = build_run_args(&cfg, "t1", Path::new("/tmp/rt"));
        for flag in [
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--memory=512m",
            "--pids-limit=256",
            "--timeout=300",
        ] {
            assert!(has(&a, flag), "missing {flag}: {a:?}");
        }
        assert!(has(&a, "/tmp/ws:/workspace"));
        assert!(has(&a, "/tmp/rt:/opt/agent:ro"));
        assert_eq!(&a[a.len() - 2..], ["sleep", "infinity"]);
    }

    #[test]
    fn network_and_skills_are_opt_in() {
        let mut cfg = PodmanConfig::new("/tmp/ws");
        cfg.network = NetworkPolicy::Enabled;
        cfg.skills_dir = Some("/home/me/skills".into());
        cfg.env = vec![("MY_CONFIG".into(), "v".into())];
        let a = build_run_args(&cfg, "t2", Path::new("/tmp/rt"));
        assert!(!has(&a, "--network=none"));
        assert!(has(&a, "/home/me/skills:/skills:ro"));
        assert!(has(&a, "MY_CONFIG=v"));
    }

    #[test]
    fn docker_does_not_get_the_podman_only_timeout_flag() {
        let mut cfg = PodmanConfig::new("/tmp/ws");
        cfg.binary = "docker".into();
        assert!(
            !build_run_args(&cfg, "t3", Path::new("/tmp/rt"))
                .iter()
                .any(|a| a.starts_with("--timeout"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_inside_the_workspace_are_rejected() {
        let ws = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), ws.path().join("link")).unwrap();
        assert!(reject_symlinks(ws.path(), Path::new("link/evil.txt")).is_err());
        assert!(reject_symlinks(ws.path(), Path::new("fine/evil.txt")).is_ok());
    }

    #[test]
    fn write_into_workspace_writes_inside_the_root() {
        let ws = tempfile::tempdir().unwrap();
        let p = crate::sandbox::WorkspacePath::parse("out/a.csv").unwrap();
        write_into_workspace(ws.path(), &p, b"x,y\n").unwrap();
        assert_eq!(
            std::fs::read(ws.path().join("out/a.csv")).unwrap(),
            b"x,y\n"
        );
    }

    #[cfg(unix)]
    #[test]
    fn write_into_workspace_refuses_to_follow_a_symlink() {
        let ws = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), ws.path().join("link")).unwrap();
        let p = crate::sandbox::WorkspacePath::parse("link/evil.txt").unwrap();
        assert!(write_into_workspace(ws.path(), &p, b"x").is_err());
        assert!(!outside.path().join("evil.txt").exists());
    }

    #[tokio::test]
    #[ignore = "needs podman and the ai-agent-sandbox image"]
    async fn state_persists_across_executions() {
        use crate::sandbox::CodeExecutor;
        let sandbox = PodmanSandbox::start(PodmanConfig::new("./target/test-ws"))
            .await
            .unwrap();

        let r1 = sandbox
            .run_python("data = [1, 2, 3]; print('ok')")
            .await
            .unwrap();
        assert_eq!(r1.stdout.trim(), "ok");

        let r2 = sandbox.run_python("sum(data)").await.unwrap();
        assert_eq!(r2.value.as_deref(), Some("6"));

        let r3 = sandbox.run_python("x = 1 / 0").await.unwrap();
        assert_eq!(r3.error.unwrap().name, "ZeroDivisionError");
        let r4 = sandbox.run_python("sum(data)").await.unwrap();
        assert_eq!(r4.value.as_deref(), Some("6"));

        sandbox.shutdown().await.unwrap();
    }

    #[tokio::test]
    #[ignore = "needs podman and the ai-agent-sandbox image"]
    async fn sandbox_code_can_call_host_tools() {
        struct Double;

        #[async_trait]
        impl SandboxTool for Double {
            fn name(&self) -> &str {
                "double"
            }
            fn definition(&self) -> serde_json::Value {
                serde_json::json!({"name": "double", "description": "Doubles x."})
            }
            async fn call(&self, args: serde_json::Value) -> Result<serde_json::Value, String> {
                Ok(serde_json::json!(args["x"].as_i64().unwrap_or(0) * 2))
            }
        }

        let sandbox = PodmanSandbox::start_with_tools(
            PodmanConfig::new("./target/test-ws-bridge"),
            vec![Arc::new(Double)],
        )
        .await
        .unwrap();
        let r = sandbox.run_python("double(x=21)").await.unwrap();
        assert_eq!(r.value.as_deref(), Some("42"));
        sandbox.shutdown().await.unwrap();
    }
}

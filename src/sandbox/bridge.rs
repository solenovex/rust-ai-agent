use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

/// A host-side tool callable from sandboxed Python as a plain function.
/// It has no `ExecutionContext` parameter, so it cannot depend on agent state.
#[async_trait]
pub trait SandboxTool: Send + Sync {
    fn name(&self) -> &str;
    /// JSON definition (name, description, parameters) shown to the LLM.
    fn definition(&self) -> Value;
    async fn call(&self, args: Value) -> Result<Value, String>;
}

/// Generates the Python stub injected into the sandbox. The stub forwards
/// keyword arguments over the bridge; the real work runs on the host.
pub fn python_stub(tool: &dyn SandboxTool) -> String {
    let desc = tool.definition()["description"]
        .as_str()
        .unwrap_or("")
        .replace("\"\"\"", "'''")
        .replace('\\', "/");
    let name = tool.name();
    format!(
        "def {name}(**kwargs):\n    \"\"\"{desc}\"\"\"\n    return _call_tool(\"{name}\", kwargs)\n"
    )
}

/// Describes the sandbox-callable functions for the system prompt.
pub fn sandbox_tools_prompt(tools: &[Arc<dyn SandboxTool>]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let defs: Vec<Value> = tools.iter().map(|t| t.definition()).collect();
    let json = serde_json::to_string_pretty(&defs).expect("serializable");
    format!(
        "\n\n## Sandbox-Executable Tools\nThe following functions are pre-registered in the sandbox \
         and can be called directly in your Python code (keyword arguments only):\n{json}"
    )
}

/// Handles one `tool_call` message from the runner and builds the reply.
pub async fn dispatch(tools: &HashMap<String, Arc<dyn SandboxTool>>, msg: &Value) -> Value {
    let name = msg["name"].as_str().unwrap_or_default();
    match tools.get(name) {
        Some(tool) => match tool.call(msg["args"].clone()).await {
            Ok(v) => json!({ "result": v }),
            Err(e) => json!({ "error": e }),
        },
        None => json!({ "error": format!("unknown tool: {name}") }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;

    #[async_trait]
    impl SandboxTool for Echo {
        fn name(&self) -> &str {
            "echo_tool"
        }
        fn definition(&self) -> Value {
            json!({"name": "echo_tool", "description": "Echo back the arguments."})
        }
        async fn call(&self, args: Value) -> Result<Value, String> {
            Ok(args)
        }
    }

    #[test]
    fn python_stub_forwards_kwargs_to_the_bridge() {
        let stub = python_stub(&Echo);
        assert!(stub.starts_with("def echo_tool(**kwargs):"), "{stub}");
        assert!(stub.contains("_call_tool(\"echo_tool\", kwargs)"), "{stub}");
    }

    #[test]
    fn prompt_is_empty_without_tools_and_lists_them_otherwise() {
        assert_eq!(sandbox_tools_prompt(&[]), "");
        let t: Arc<dyn SandboxTool> = Arc::new(Echo);
        assert!(sandbox_tools_prompt(&[t]).contains("echo_tool"));
    }

    #[tokio::test]
    async fn dispatch_returns_the_tool_result() {
        let mut tools: HashMap<String, Arc<dyn SandboxTool>> = HashMap::new();
        tools.insert("echo_tool".into(), Arc::new(Echo));
        let reply = dispatch(
            &tools,
            &json!({"type": "tool_call", "name": "echo_tool", "args": {"x": 1}}),
        )
        .await;
        assert_eq!(reply, json!({"result": {"x": 1}}));
    }

    #[tokio::test]
    async fn dispatch_reports_unknown_tools_as_errors() {
        let tools: HashMap<String, Arc<dyn SandboxTool>> = HashMap::new();
        let reply = dispatch(
            &tools,
            &json!({"type": "tool_call", "name": "nope", "args": {}}),
        )
        .await;
        assert_eq!(reply["error"], "unknown tool: nope");
    }
}

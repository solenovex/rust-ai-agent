//! 把 [`Agent`] 适配成普通 [`Tool`] 的包装器。
//!
//! 这就是 "Agent as Tool" 模式：协调员的 `ToolBox` 仍是普通工具表，只不过
//! 其中某些条目本身是另一个 Agent。从 LLM 的视角看，agent 工具和
//! `web_search`、`calculator` 没有区别——都有名称、描述和参数，调用后返回
//! 一个字符串。
//!
//! 子代理在**全新的 session** 中运行，所以协调员只能看到子代理的最终答案，
//! 看不到它中间的工具调用和推理过程。

use std::sync::Arc;

use schemars::{JsonSchema, schema_for};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::{
    agent::{
        context::ExecutionContext,
        model::{Agent, AgentStatus},
        runtime::push_user_message,
        tool_confirmation::ToolConfirmation,
    },
    tool::Tool,
};

/// 包装一个 Agent，使其可以注册到另一个 Agent 的 `ToolBox` 中。
///
/// 默认工具参数只有一个 `request` 字符串。使用
/// [`AgentTool::with_input_schema`] 可以改成结构化参数。
pub struct AgentTool {
    agent: Arc<Agent>,
    name: String,
    description: String,
    parameters: Value,
    /// `true` 表示 `parameters` 来自带类型的 `input_schema`。
    structured_input: bool,
    requires_confirmation: bool,
    /// 自动批准子代理内部那些需要确认的工具。
    auto_approve_child_tools: bool,
}

impl AgentTool {
    /// 用默认的 `{ "request": string }` 参数包装 `agent`。
    pub fn new(agent: Arc<Agent>) -> Self {
        let description = if agent.description.is_empty() {
            format!("Delegate a task to the '{}' agent.", agent.name)
        } else {
            agent.description.clone()
        };

        Self {
            name: agent.name.clone(),
            description,
            agent,
            parameters: json!({
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The task or question to delegate to this agent"
                    }
                },
                "required": ["request"]
            }),
            structured_input: false,
            requires_confirmation: false,
            auto_approve_child_tools: false,
        }
    }

    /// 使用带类型的 schema 作为工具参数。
    ///
    /// 协调员传入的参数会先经过 `serde` 校验，然后以 JSON 字符串的形式发给
    /// 子代理，子代理可以自行解析其中的各个字段。
    pub fn with_input_schema<T: JsonSchema>(mut self) -> Self {
        let mut parameters = serde_json::to_value(schema_for!(T))
            .expect("JSON schema for AgentTool input is always serializable");
        // schemars 会生成一个顶层 title，对模型没有用，去掉。
        if let Some(object) = parameters.as_object_mut() {
            object.remove("title");
        }
        self.parameters = parameters;
        self.structured_input = true;
        self
    }

    /// 在子代理运行前，先让**父级**用户确认是否委派。
    ///
    /// 对能力很强的代理（例如可以执行 shell 的 coder）很有用。默认 `false`。
    pub fn with_requires_confirmation(mut self, required: bool) -> Self {
        self.requires_confirmation = required;
        self
    }

    /// 自动批准子代理想要调用的、需要确认的工具。
    ///
    /// 子代理是自主运行的；否则 `execute_python` 这类工具会停在
    /// `PendingConfirmation`，而现场没有用户能点同意。只对信任其工具的代理
    /// 开启这个选项。
    pub fn with_auto_approve_child_tools(mut self, auto_approve: bool) -> Self {
        self.auto_approve_child_tools = auto_approve;
        self
    }

    /// 在全新的 session 中运行子代理，只返回它的输出。
    async fn run_child(&self, request: String) -> anyhow::Result<String> {
        // 上下文隔离：新的 session id 意味着父级事件对子代理不可见，
        // 子代理的事件也不会回流到协调员的上下文里。
        let session_id = format!("{}-{}", self.agent.name, Uuid::new_v4());
        let session = self
            .agent
            .session_manager
            .get_or_create(&session_id, None)
            .await?;
        let mut context = ExecutionContext::new(session);
        push_user_message(&mut context, &request);

        loop {
            let result = self.agent.run_in_context(context).await?;

            match result.status {
                AgentStatus::Complete => {
                    return match result.output {
                        Some(Value::String(text)) => Ok(text),
                        Some(other) => Ok(other.to_string()),
                        None => anyhow::bail!(
                            "agent '{}' completed without producing an output",
                            self.agent.name
                        ),
                    };
                }
                AgentStatus::PendingConfirmation if self.auto_approve_child_tools => {
                    let confirmations: Vec<ToolConfirmation> = result
                        .context
                        .session
                        .pending_tool_calls
                        .iter()
                        .map(|pending| ToolConfirmation {
                            tool_call_id: pending.tool_call.id.clone(),
                            reason: Some("auto-approved by AgentTool".to_owned()),
                            approved: true,
                            modified_arguments: None,
                        })
                        .collect();

                    tracing::info!(
                        "AgentTool '{}': auto-approving {} child tool call(s)",
                        self.name,
                        confirmations.len()
                    );

                    // 下一步 `prepare_llm_request` 会消费
                    // `pending_tool_calls` 和 `state["tool_confirmations"]`。
                    context = result.context;
                    context.session.state.insert(
                        "tool_confirmations".to_owned(),
                        serde_json::to_value(&confirmations)?,
                    );
                }
                AgentStatus::PendingConfirmation => {
                    anyhow::bail!(
                        "agent '{}' needs tool confirmation; enable \
                         with_auto_approve_child_tools(true) or ask the parent user first",
                        self.agent.name
                    );
                }
                AgentStatus::Error => {
                    anyhow::bail!("agent '{}' finished with an error", self.agent.name);
                }
            }
        }
    }
}

#[async_trait::async_trait]
impl Tool for AgentTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        // 先解析参数，这样非法 JSON 会作为工具错误返回，而不是被当成文本
        // 原样转给子代理。
        let arguments: Value = serde_json::from_str(args_json)?;

        let request = if self.structured_input {
            // 把结构化字段保留成 JSON，交给子代理解析。
            serde_json::to_string(&arguments)?
        } else {
            match arguments.get("request").and_then(Value::as_str) {
                Some(request) => request.to_owned(),
                None => anyhow::bail!(
                    "tool '{}' expects a string parameter named 'request'",
                    self.name
                ),
            }
        };

        tracing::info!("AgentTool '{}' delegating: {request}", self.name);
        self.run_child(request).await
    }

    fn requires_confirmation(&self) -> bool {
        self.requires_confirmation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolBox;

    fn child() -> Arc<Agent> {
        Arc::new(
            Agent::new(
                "test-model".to_owned(),
                vec!["You are a specialist.".to_owned()],
                Arc::new(ToolBox::new()),
            )
            .with_name("researcher".to_owned())
            .with_description("Research topics on the web".to_owned()),
        )
    }

    #[test]
    fn default_parameters_are_a_single_request_string() {
        let tool = AgentTool::new(child());
        assert_eq!(tool.name(), "researcher");
        assert_eq!(tool.description(), "Research topics on the web");
        assert_eq!(tool.parameters()["required"], json!(["request"]));
        assert_eq!(tool.parameters()["properties"]["request"]["type"], "string");
        assert!(!tool.requires_confirmation());
    }

    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct ResearchRequest {
        /// 要调研的主题。
        topic: String,
        /// 最多参考多少个来源。
        max_sources: u8,
    }

    #[test]
    fn structured_schema_replaces_the_request_parameter() {
        let tool = AgentTool::new(child()).with_input_schema::<ResearchRequest>();
        let parameters = tool.parameters();
        assert_eq!(parameters["type"], "object");
        assert!(parameters["properties"].get("topic").is_some());
        assert!(parameters["properties"].get("max_sources").is_some());
        assert!(parameters.get("title").is_none());
    }

    #[test]
    fn description_falls_back_to_a_delegation_hint() {
        let agent = Arc::new(Agent::new(
            "test-model".to_owned(),
            vec![],
            Arc::new(ToolBox::new()),
        ));
        let tool = AgentTool::new(agent);
        assert!(tool.description().contains("Delegate a task"));
    }
}

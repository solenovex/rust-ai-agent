//! Transfer 模式使用的 `transfer_to_agent` 工具。
//!
//! 从 LLM 的视角看，它就是一个普通工具：有名称、描述和参数 schema。
//! 但它的副作用不是执行业务操作，而是**请求把控制权交给另一个 agent**。
//!
//! 与书中的 Python 实现不同，这里的 [`Tool::execute`] 不直接修改
//! `ExecutionContext`：`Tool::execute` 接收的是只读上下文，而 Rust 更适合把
//! “修改执行状态”这件事集中到运行时里完成。因此真正的状态切换在
//! `Agent::act_all` 中识别到本工具后执行，见
//! [`crate::agent::runtime::act`]。

use std::sync::Arc;

use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    agent::{context::ExecutionContext, model::Agent},
    constant::TRANSFER_TOOL_NAME,
    tool::Tool,
};

/// `transfer_to_agent` 的参数。
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TransferArgs {
    /// 目标 agent 的名称，必须是当前 agent 可转移目标之一。
    pub agent_name: String,
}

/// 动态生成到 LLM 请求里的转移工具。
///
/// 每个 agent 能转移到的目标不同，所以工具要在请求时根据 agent 树动态生成。
/// 这里刻意不缓存：agent 建立 parent 关系后，它的兄弟节点才变得可见，
/// 因此“按需生成”比在构造时缓存更不容易产生过期数据。
pub struct TransferTool {
    targets: Vec<Arc<Agent>>,
    target_names: Vec<String>,
    description: String,
    parameters: Value,
}

impl TransferTool {
    /// 根据当前 agent 允许转移到的目标创建工具。
    ///
    /// `targets` 为空时不应该调用本函数；调用方应先检查，
    /// 因为没有目标时不需要向 LLM 暴露转移工具。
    pub fn new(targets: Vec<Arc<Agent>>) -> Self {
        let target_names: Vec<String> = targets.iter().map(|agent| agent.name.clone()).collect();

        let mut agent_lines = Vec::with_capacity(targets.len());
        for agent in &targets {
            agent_lines.push(format!(
                "  - {}: {}",
                agent.name,
                summarize_agent(&agent.description, &agent.instructions)
            ));
        }

        let description = format!(
            "Transfer control of the current conversation to another agent.\n\n\
             Use this tool when the current request clearly belongs to another agent's \
             specialty. The full conversation history is preserved, and the selected \
             agent will answer the user directly. Do not use it for tasks you can \
             handle yourself.\n\n\
             Available agents:\n{}",
            agent_lines.join("\n")
        );

        // 用 serde 生成基础 schema，再动态补上 enum 约束。
        // 这样参数的 `type`、`required` 等部分不必手写，也不容易写错。
        let mut parameters = serde_json::to_value(schema_for!(TransferArgs))
            .expect("TransferArgs schema is always serializable");
        if let Some(object) = parameters.as_object_mut() {
            object.remove("title");
        }
        parameters["properties"]["agent_name"]["enum"] = json!(target_names);
        parameters["properties"]["agent_name"]["description"] =
            json!("Name of the agent to transfer to.");

        Self {
            targets,
            target_names,
            description,
            parameters,
        }
    }

    /// 当前工具允许的目标名称，按注册顺序返回。
    pub fn target_names(&self) -> &[String] {
        &self.target_names
    }

    /// 解析并校验参数，返回目标 agent。
    ///
    /// 运行时收到的是 LLM 生成的 JSON，不能假设 enum 一定生效，
    /// 所以这里必须再校验一次。
    pub fn resolve(&self, args_json: &str) -> anyhow::Result<Arc<Agent>> {
        let args: TransferArgs = serde_json::from_str(args_json)
            .map_err(|err| anyhow::anyhow!("invalid transfer arguments: {err}"))?;

        self.targets
            .iter()
            .find(|agent| agent.name == args.agent_name)
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "'{}' is not a valid transfer target; available: {:?}",
                    args.agent_name,
                    self.target_names
                )
            })
    }
}

#[async_trait::async_trait]
impl Tool for TransferTool {
    fn name(&self) -> &str {
        TRANSFER_TOOL_NAME
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    fn requires_confirmation(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        args_json: &str,
        _context: &ExecutionContext,
    ) -> anyhow::Result<String> {
        // 这个返回值只是为了保持 Tool 接口完整。真正的转移由
        // `Agent::act_all` 拦截并写入 ExecutionContext::transfer_to。
        let target = self.resolve(args_json)?;
        Ok(format!("Transferring to {}...", target.name))
    }
}

/// 给工具描述生成一段简短的 agent 介绍。
fn summarize_agent(description: &str, instructions: &[String]) -> String {
    let text = if description.trim().is_empty() {
        instructions.join(" ")
    } else {
        description.to_owned()
    };
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");

    const MAX_CHARS: usize = 100;
    if text.chars().count() > MAX_CHARS {
        format!("{}...", text.chars().take(MAX_CHARS).collect::<String>())
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use async_openai::types::chat::{
        ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls, FunctionCall,
    };

    use super::*;
    use crate::tool::ToolBox;

    fn agent(name: &str, description: &str) -> Arc<Agent> {
        Arc::new(
            Agent::new(
                "test-model".to_owned(),
                vec![format!("You are the {name} agent.")],
                Arc::new(ToolBox::new()),
            )
            .with_name(name.to_owned())
            .with_description(description.to_owned()),
        )
    }

    fn router_with_children() -> (Arc<Agent>, Arc<Agent>, Arc<Agent>) {
        let math = agent("math", "Math calculation specialist");
        let search = agent("search", "Information search specialist");
        let router = agent("router", "Routes requests to specialists");
        router
            .set_sub_agents(vec![math.clone(), search.clone()])
            .unwrap();
        (router, math, search)
    }

    #[test]
    fn tool_definition_contains_descriptions_and_enum() {
        let (router, _, _) = router_with_children();
        let tool = TransferTool::new(router.transfer_targets());
        let parameters = tool.parameters();

        assert_eq!(tool.name(), TRANSFER_TOOL_NAME);
        assert!(tool.description().contains("Math calculation specialist"));
        assert_eq!(
            parameters["properties"]["agent_name"]["enum"],
            json!(["math", "search"])
        );
        assert_eq!(parameters["required"], json!(["agent_name"]));
    }

    #[test]
    fn resolve_accepts_only_allowed_targets() {
        let (router, _, _) = router_with_children();
        let tool = TransferTool::new(router.transfer_targets());

        let target = tool.resolve(r#"{"agent_name":"math"}"#).unwrap();
        assert_eq!(target.name, "math");

        let err = match tool.resolve(r#"{"agent_name":"nobody"}"#) {
            Ok(_) => panic!("unexpected transfer target"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("not a valid transfer target"));
    }

    #[tokio::test]
    async fn execute_reports_the_target_without_side_effects() {
        let (router, _, _) = router_with_children();
        let tool = TransferTool::new(router.transfer_targets());
        let context = ExecutionContext::default();

        let result = tool
            .execute(r#"{"agent_name":"search"}"#, &context)
            .await
            .unwrap();
        assert_eq!(result, "Transferring to search...");
    }

    fn transfer_call(id: &str, target: &str) -> ChatCompletionMessageToolCalls {
        ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
            id: id.to_owned(),
            function: FunctionCall {
                name: TRANSFER_TOOL_NAME.to_owned(),
                arguments: json!({ "agent_name": target }).to_string(),
            },
        })
    }

    #[tokio::test]
    async fn act_all_records_only_the_first_valid_transfer() {
        let (router, _, _) = router_with_children();
        let mut context = ExecutionContext::default();

        router
            .act_all(
                &[
                    transfer_call("call_1", "math"),
                    transfer_call("call_2", "search"),
                ],
                &mut context,
            )
            .await
            .unwrap();

        assert_eq!(context.transfer_to.as_deref(), Some("math"));
        assert_eq!(
            context
                .session
                .events
                .iter()
                .filter(|event| {
                    event.content.iter().any(|item| {
                        matches!(item, crate::agent::event::ContentItem::ToolResult(result)
                        if result.name == TRANSFER_TOOL_NAME)
                    })
                })
                .count(),
            2,
            "both tool calls need a tool result so the conversation stays valid"
        );
    }

    #[tokio::test]
    async fn act_all_rejects_an_unknown_target() {
        let (router, _, _) = router_with_children();
        let mut context = ExecutionContext::default();

        router
            .act_all(&[transfer_call("call_1", "nobody")], &mut context)
            .await
            .unwrap();

        assert!(context.transfer_to.is_none());
        let has_error = context.session.events.iter().any(|event| {
            event.content.iter().any(|item| {
                matches!(item, crate::agent::event::ContentItem::ToolResult(result)
                    if result.name == TRANSFER_TOOL_NAME
                        && matches!(result.status, crate::agent::event::ToolResultStatus::Error))
            })
        });
        assert!(has_error);
    }
}

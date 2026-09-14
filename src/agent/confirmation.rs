use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::{
    context::ExecutionContext,
    event::{ContentItem, Event, ToolCall, ToolResultStatus},
    llm_request::LlmRequest,
    runtime::Agent,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingToolCall {
    pub confirmation_message: String,
    pub tool_call: ToolCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfirmation {
    pub tool_call_id: String,
    pub reason: Option<String>,
    pub approved: bool,
    pub modified_arguments: Option<serde_json::Map<String, serde_json::Value>>,
}

impl Agent {
    /// Builds the next LLM request, first resolving any tool calls that were
    /// paused pending human confirmation on a previous `run()` invocation.
    pub(super) async fn prepare_llm_request(
        &self,
        context: &mut ExecutionContext,
    ) -> anyhow::Result<LlmRequest> {
        if let (Some(raw_pending), Some(raw_confirmations)) = (
            context.state_mut().remove("pending_tool_calls"),
            context.state_mut().remove("tool_confirmations"),
        ) {
            let confirmation_results = self
                .process_confirmations(context, raw_pending, raw_confirmations)
                .await?;

            if !confirmation_results.is_empty() {
                context.add_event(Event::new(
                    context.execution_id.clone(),
                    "tool",
                    confirmation_results,
                ));
            }
        }

        let mut request = LlmRequest {
            instructions: Vec::new(),
            contents: context
                .events()
                .iter()
                .flat_map(|ev| ev.content.iter().cloned())
                .collect(),
        };

        for cb in &self.before_llm_callbacks {
            cb.call(context, &mut request).await;
        }

        Ok(request)
    }

    /// Consumes a paused `PendingToolCall` batch together with the matching
    /// `ToolConfirmation`s and either executes each approved tool call or
    /// records why it was not run.
    async fn process_confirmations(
        &self,
        context: &ExecutionContext,
        raw_pending: Value,
        raw_confirmations: Value,
    ) -> anyhow::Result<Vec<ContentItem>> {
        let pending_calls: Vec<PendingToolCall> = serde_json::from_value(raw_pending)?;
        let confirmations: Vec<ToolConfirmation> = serde_json::from_value(raw_confirmations)?;

        let confirmation_map: HashMap<String, ToolConfirmation> = confirmations
            .into_iter()
            .map(|c| (c.tool_call_id.clone(), c))
            .collect();

        let mut results = Vec::with_capacity(pending_calls.len());

        for pending in pending_calls {
            let tool_call_id = pending.tool_call.tool_call_id;
            let name = pending.tool_call.name;
            let confirmation = confirmation_map.get(&tool_call_id);

            let (status, content) = match confirmation {
                Some(c) if c.approved => {
                    let mut arguments = pending
                        .tool_call
                        .arguments
                        .as_object()
                        .cloned()
                        .unwrap_or_default();
                    if let Some(overrides) = &c.modified_arguments {
                        arguments.extend(overrides.clone());
                    }
                    let args_json = Value::Object(arguments).to_string();

                    match self.toolbox.get(&name) {
                        Some(tool) => match tool.execute(&args_json, context).await {
                            Ok(result) => (ToolResultStatus::Success, result),
                            Err(err) => (
                                ToolResultStatus::Error,
                                format!("Tool execution error: {err}"),
                            ),
                        },
                        None => (
                            ToolResultStatus::Error,
                            format!("Tool execution error: unknown tool {name}"),
                        ),
                    }
                }
                Some(c) => {
                    let reason = c
                        .reason
                        .clone()
                        .unwrap_or_else(|| "Tool execution was rejected by user.".to_string());
                    (ToolResultStatus::Error, reason)
                }
                None => (
                    ToolResultStatus::Error,
                    "Tool execution was not approved.".to_string(),
                ),
            };

            results.push(ContentItem::ToolResult {
                tool_call_id,
                name,
                status,
                content,
            });
        }

        Ok(results)
    }
}

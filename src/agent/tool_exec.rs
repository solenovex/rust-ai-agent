use async_openai::types::chat::ChatCompletionMessageToolCalls;
use serde_json::Value;

use crate::agent::{
    callback::ToolCallView,
    confirmation::PendingToolCall,
    context::ExecutionContext,
    event::{ContentItem, Event, ToolCall, ToolResultStatus},
    runtime::Agent,
};

impl Agent {
    pub(super) fn record_tool_calls(
        &self,
        context: &mut ExecutionContext,
        tool_calls: &[ChatCompletionMessageToolCalls],
    ) {
        let mut call_items = Vec::new();
        for tool_call in tool_calls {
            if let ChatCompletionMessageToolCalls::Function(function_call) = tool_call {
                let arguments: serde_json::Value =
                    serde_json::from_str(&function_call.function.arguments)
                        .unwrap_or(serde_json::Value::Null);
                call_items.push(ContentItem::ToolCall(ToolCall {
                    tool_call_id: function_call.id.clone(),
                    name: function_call.function.name.clone(),
                    arguments,
                }));
            }
        }
        context.add_event(Event::new(
            context.execution_id.clone(),
            "agent",
            call_items,
        ));
    }

    pub(super) async fn execute_tool_calls(
        &self,
        context: &mut ExecutionContext,
        tool_calls: &[ChatCompletionMessageToolCalls],
    ) {
        let mut result_items = Vec::new();
        let mut pending_tool_calls: Vec<PendingToolCall> = Vec::new();

        for tool_call in tool_calls {
            let ChatCompletionMessageToolCalls::Function(function_call) = tool_call else {
                continue;
            };
            let function_name = &function_call.function.name;
            let arguments = &function_call.function.arguments;

            tracing::info!("Tool call: {function_name}({arguments})");

            let arguments_value: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);

            if let Some(tool) = self.toolbox.get(function_name)
                && tool.requires_confirmation() {
                    tracing::info!("Tool call requires confirmation: {function_name}");
                    pending_tool_calls.push(PendingToolCall {
                        tool_call: ToolCall {
                            tool_call_id: function_call.id.clone(),
                            name: function_name.clone(),
                            arguments: arguments_value.clone(),
                        },
                        confirmation_message: tool.get_confirmation_message(&arguments_value),
                    });
                    continue;
                }

            let view = ToolCallView {
                tool_call_id: &function_call.id,
                name: function_name,
                arguments: &arguments_value,
            };

            let mut short_circuited = None;
            for callback in &self.before_tool_callbacks {
                if let Some(result) = callback.call(context, view).await {
                    short_circuited = Some(result);
                    break;
                }
            }

            let (mut status, mut content) = match short_circuited {
                Some(result) => (ToolResultStatus::Success, result),
                None => match self.toolbox.get(function_name) {
                    Some(tool) => match tool.execute(arguments, context).await {
                        Ok(result) => {
                            tracing::info!("Tool result: {result}");
                            (ToolResultStatus::Success, result)
                        }
                        Err(err) => {
                            let msg = format!("Tool execution error: {err}");
                            tracing::error!("{msg}");
                            (ToolResultStatus::Error, msg)
                        }
                    },
                    None => {
                        let msg = format!("Tool execution error: unknown tool {function_name}");
                        tracing::error!("{msg}");
                        (ToolResultStatus::Error, msg)
                    }
                },
            };

            for callback in &self.after_tool_callbacks {
                if let Some((new_status, new_content)) = callback
                    .call(context, &function_call.id, function_name, status, &content)
                    .await
                {
                    status = new_status;
                    content = new_content;
                    break;
                }
            }

            result_items.push(ContentItem::ToolResult {
                tool_call_id: function_call.id.clone(),
                name: function_name.clone(),
                status,
                content,
            });
        }

        if !pending_tool_calls.is_empty() {
            let value = serde_json::to_value(&pending_tool_calls).unwrap_or(Value::Null);
            context
                .state_mut()
                .insert("pending_tool_calls".to_string(), value);
        }

        context.add_event(Event::new(
            context.execution_id.clone(),
            "tool",
            result_items,
        ));
    }
}

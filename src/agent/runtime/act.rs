use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls, ChatCompletionRequestMessage,
    ChatCompletionRequestToolMessageArgs,
};

use crate::agent::{
    context::ExecutionContext,
    event::{AgentEvent, ContentItem, ToolCall, ToolResult, ToolResultStatus},
    model::Agent,
    tool_confirmation::PendingToolCall,
};

impl Agent {
    pub async fn act_all(
        &self,
        tool_calls: &[ChatCompletionMessageToolCalls],
        context: &mut ExecutionContext,
    ) -> anyhow::Result<()> {
        for tool_call in tool_calls {
            match tool_call {
                ChatCompletionMessageToolCalls::Function(tool_call) => {
                    if let Some(tool) = self.toolbox.get(&tool_call.function.name)
                        && tool.requires_confirmation()
                    {
                        let arguments = serde_json::from_str(&tool_call.function.arguments)
                            .unwrap_or(serde_json::Value::Null);
                        context.session.add_event(AgentEvent::new(
                            context.execution_id.clone(),
                            "assistant",
                            vec![ContentItem::ToolCall(ToolCall::new(
                                tool_call.id.clone(),
                                tool_call.function.name.clone(),
                                arguments.clone(),
                            ))],
                        ));
                        context.session.add_pending_tool_call(PendingToolCall::new(
                            tool_call.clone(),
                            tool.get_confirmation_message(&arguments),
                        ));
                        continue;
                    }
                    self.act(tool_call, context).await?;
                }
                ChatCompletionMessageToolCalls::Custom(custom_tool_call) => {
                    tracing::info!("Custom tool call not implemented");
                    context.session.add_event(AgentEvent::new(
                        context.execution_id.clone(),
                        "agent",
                        vec![ContentItem::ToolResult(ToolResult::new(
                            custom_tool_call.id.clone(),
                            custom_tool_call.custom_tool.name.clone(),
                            ToolResultStatus::Error,
                            "Custom tool call not implemented".to_owned(),
                        ))],
                    ));
                }
            }
        }

        Ok(())
    }

    async fn act(
        &self,
        tool_call: &ChatCompletionMessageToolCall,
        context: &mut ExecutionContext,
    ) -> anyhow::Result<ChatCompletionRequestMessage> {
        let tool_call_id = tool_call.id.clone();
        let tool_name = tool_call.function.name.clone();
        let args_json = tool_call.function.arguments.clone();

        context.session.add_event(AgentEvent::new(
            context.execution_id.clone(),
            "assistant",
            vec![ContentItem::ToolCall(ToolCall::new(
                tool_call_id.clone(),
                tool_name.clone(),
                serde_json::from_str(&args_json)?,
            ))],
        ));

        tracing::info!("Tool call request ({tool_call_id}): {tool_name} with args: {args_json}");

        let tool = self.toolbox.get(&tool_name);

        let mut before_callback_result = None;
        for callback in self.before_tool_callbacks.iter() {
            before_callback_result = callback.call(context, tool_call).await;
            if before_callback_result.is_some() {
                break;
            }
        }

        if let Some(before_callback_result) = before_callback_result {
            context.session.add_event(AgentEvent::new(
                context.execution_id.clone(),
                "agent",
                vec![ContentItem::ToolResult(ToolResult::new(
                    tool_call_id.clone(),
                    tool_name.clone(),
                    ToolResultStatus::Success,
                    before_callback_result.clone(),
                ))],
            ));

            Ok(ChatCompletionRequestToolMessageArgs::default()
                .tool_call_id(tool_call_id.clone())
                .content(before_callback_result)
                .build()?
                .into())
        } else {
            let tool_result = match tool {
                Some(tool) => tool.execute(&args_json, context).await,
                None => Ok(format!(
                    "Tool error: unable to get tool: {tool_name} from ToolBox."
                )),
            };

            let (status, content) = match tool_result {
                Ok(result) => {
                    tracing::info!("Tool call result ({tool_call_id}): {result}");
                    (ToolResultStatus::Success, result)
                }
                Err(err) => {
                    let msg = format!("Tool call failed ({tool_call_id}): {}", err);
                    tracing::info!("{msg}");
                    (ToolResultStatus::Error, msg)
                }
            };

            let mut after_callback_result = None;
            for callback in self.after_tool_callbacks.iter() {
                after_callback_result = callback.call(context, tool_call, &content).await;
                if after_callback_result.is_some() {
                    break;
                }
            }

            let result = after_callback_result.unwrap_or(content);

            context.session.add_event(AgentEvent::new(
                context.execution_id.clone(),
                "agent",
                vec![ContentItem::ToolResult(ToolResult::new(
                    tool_call_id.clone(),
                    tool_name.clone(),
                    status,
                    result.clone(),
                ))],
            ));

            Ok(ChatCompletionRequestToolMessageArgs::default()
                .tool_call_id(tool_call_id.clone())
                .content(result.clone())
                .build()?
                .into())
        }
    }
}

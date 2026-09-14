use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionTools, FunctionCall,
    FunctionObjectArgs,
};

use crate::agent::{event::ToolCall, runtime::Agent, ContentItem, ExecutionContext};

#[derive(Debug, Clone, Default)]
pub struct LlmRequest {
    pub instructions: Vec<String>,
    pub contents: Vec<ContentItem>,
}

impl LlmRequest {
    pub fn append_instructions(&mut self, instruction: impl Into<String>) {
        self.instructions.push(instruction.into());
    }
}

#[async_trait::async_trait]
pub trait BeforeLlmCallback: Send + Sync {
    async fn call(&self, context: &mut ExecutionContext, request: &mut LlmRequest);
}

impl Agent {
    /// Translates the accumulated `ContentItem`s in an `LlmRequest` into the
    /// `async-openai` chat message types, merging system instructions,
    /// consecutive assistant tool calls, and tool results.
    pub(super) fn build_messages(
        &self,
        request: &LlmRequest,
    ) -> anyhow::Result<Vec<ChatCompletionRequestMessage>> {
        let mut messages = Vec::new();

        if let Some(system) = &self.instructions {
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system.as_str())
                    .build()?
                    .into(),
            );
        }

        for extra_instruction in &request.instructions {
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(extra_instruction.as_str())
                    .build()?
                    .into(),
            );
        }

        for item in &request.contents {
            match item {
                ContentItem::Message { role, content } => {
                    let message: ChatCompletionRequestMessage = if role == "user" {
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(content.clone())
                            .build()?
                            .into()
                    } else {
                        ChatCompletionRequestAssistantMessageArgs::default()
                            .content(content.clone())
                            .build()?
                            .into()
                    };
                    messages.push(message);
                }
                ContentItem::ToolCall(ToolCall {
                    tool_call_id,
                    name,
                    arguments,
                }) => {
                    let tool_call =
                        ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                            id: tool_call_id.clone(),
                            function: FunctionCall {
                                name: name.clone(),
                                arguments: arguments.to_string(),
                            },
                        });

                    if let Some(ChatCompletionRequestMessage::Assistant(last)) = messages.last_mut()
                    {
                        last.tool_calls.get_or_insert_with(Vec::new).push(tool_call);
                    } else {
                        messages.push(
                            ChatCompletionRequestAssistantMessageArgs::default()
                                .tool_calls(vec![tool_call])
                                .build()?
                                .into(),
                        );
                    }
                }
                ContentItem::ToolResult {
                    tool_call_id,
                    content,
                    ..
                } => {
                    messages.push(
                        ChatCompletionRequestToolMessageArgs::default()
                            .tool_call_id(tool_call_id.clone())
                            .content(content.clone())
                            .build()?
                            .into(),
                    );
                }
            }
        }

        Ok(messages)
    }
}

pub(super) fn final_answer_tool_definition<T: schemars::JsonSchema>(
) -> anyhow::Result<ChatCompletionTools> {
    let schema = schemars::schema_for!(T);
    let schema_json = serde_json::to_value(&schema)?;

    let function = FunctionObjectArgs::default()
        .name("final_answer")
        .description("Return the final structured answer matching the required schema.")
        .parameters(schema_json)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build final_answer tool definition: {e}"))?;

    Ok(ChatCompletionTools::Function(ChatCompletionTool {
        function,
    }))
}
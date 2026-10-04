use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, FunctionCall,
};

use crate::{agent::event::ContentItem, tool::Tool};

pub struct LlmRequest {
    pub instructions: Vec<String>,
    pub contents: Vec<ContentItem>,
    pub tools: Option<Vec<Box<dyn Tool>>>,
    pub tool_choice: ToolChoice,
}

impl LlmRequest {
    pub fn new(instructions: Vec<String>, contents: Vec<ContentItem>) -> Self {
        Self {
            instructions,
            contents,
            tools: None,
            tool_choice: ToolChoice::default(),
        }
    }

    pub fn with_tools(mut self, tools: Vec<Box<dyn Tool>>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    pub fn append_instruction(&mut self, instruction: String) {
        self.instructions.push(instruction);
    }

    pub fn append_content(&mut self, content_item: ContentItem) {
        self.contents.push(content_item);
    }

    pub fn build_message(&self) -> anyhow::Result<Vec<ChatCompletionRequestMessage>> {
        let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();

        for instruction in self.instructions.iter() {
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(instruction.to_owned())
                    .build()?
                    .into(),
            );
        }

        for item in self.contents.iter() {
            match item {
                ContentItem::Message(message) => {
                    if message.role == "user" {
                        messages.push(
                            ChatCompletionRequestUserMessageArgs::default()
                                .content(message.content.to_owned())
                                .build()?
                                .into(),
                        );
                    } else {
                        messages.push(
                            ChatCompletionRequestAssistantMessageArgs::default()
                                .content(message.content.to_owned())
                                .build()?
                                .into(),
                        );
                    }
                }
                ContentItem::ToolCall(tool_call) => {
                    let tool_call =
                        ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                            id: tool_call.tool_call_id.clone(),
                            function: FunctionCall {
                                name: tool_call.name.clone(),
                                arguments: tool_call.arguments.to_string(),
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
                ContentItem::ToolResult(tool_result) => {
                    messages.push(
                        ChatCompletionRequestToolMessageArgs::default()
                            .tool_call_id(tool_result.tool_call_id.clone())
                            .content(tool_result.content.to_owned())
                            .build()?
                            .into(),
                    );
                }
            }
        }

        Ok(messages)
    }
}

#[derive(Debug, Default)]
pub enum ToolChoice {
    #[default]
    Auto,
    Required,
}

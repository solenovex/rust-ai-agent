use async_openai::types::chat::ChatCompletionMessageToolCall;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Clone)]
pub struct PendingToolCall {
    pub tool_call: ChatCompletionMessageToolCall,
    pub confirmation_message: String,
}

impl PendingToolCall {
    pub fn new(tool_call: ChatCompletionMessageToolCall, confirmation_message: String) -> Self {
        Self {
            tool_call,
            confirmation_message,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolConfirmation {
    pub tool_call_id: String,
    pub reason: Option<String>,
    pub approved: bool,
    pub modified_arguments: Option<ToolArguments>,
}

pub type ToolArguments = serde_json::Map<String, serde_json::Value>;

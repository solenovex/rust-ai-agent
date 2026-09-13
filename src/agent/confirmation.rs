use serde::{Deserialize, Serialize};

use crate::agent::event::ToolCall;

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

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct AgentEvent {
    pub id: String,
    pub execution_id: String,
    pub timestamp: chrono::DateTime<Utc>,
    pub author: String,
    pub content: Vec<ContentItem>,
}

impl AgentEvent {
    pub fn new(
        execution_id: impl Into<String>,
        author: impl Into<String>,
        content: Vec<ContentItem>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            execution_id: execution_id.into(),
            timestamp: chrono::Utc::now(),
            author: author.into(),
            content,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentItem {
    Message(Message),

    ToolCall(ToolCall),

    ToolResult(ToolResult),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: String, content: String) -> Self {
        Self { role, content }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub tool_call_id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    pub fn new(tool_call_id: String, name: String, arguments: serde_json::Value) -> Self {
        Self {
            tool_call_id,
            name,
            arguments,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub status: ToolResultStatus,
    pub content: String,
}

impl ToolResult {
    pub fn new(
        tool_call_id: String,
        name: String,
        status: ToolResultStatus,
        content: String,
    ) -> Self {
        Self {
            tool_call_id,
            name,
            status,
            content,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultStatus {
    Success,
    Error,
}

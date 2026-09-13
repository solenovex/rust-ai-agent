use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentItem {
    #[serde(rename = "message")]
    Message { role: String, content: String },

    #[serde(rename = "tool_call")]
    ToolCall(ToolCall),

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_call_id: String,
        name: String,
        status: ToolResultStatus,
        content: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool_call_id: String,
    pub name: String,
    pub arguments: Value,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultStatus {
    Success,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: String,
    pub execution_id: String,
    pub timestamp: i64,
    pub author: String,
    pub content: Vec<ContentItem>,
}

impl Event {
    pub fn new(
        execution_id: impl Into<String>,
        author: impl Into<String>,
        content: Vec<ContentItem>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            execution_id: execution_id.into(),
            timestamp: chrono::Utc::now().timestamp(),
            author: author.into(),
            content,
        }
    }
}
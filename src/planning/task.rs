use std::fmt::Display;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct Task {
    pub content: String,
    pub status: TaskStatus,
}

impl Display for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.status {
            TaskStatus::Pending => write!(f, "[ ] {}", self.content),
            TaskStatus::InProgress => write!(f, "[>] **{}**", self.content),
            TaskStatus::Completed => write!(f, "[x] ~~{}~~", self.content),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed
}
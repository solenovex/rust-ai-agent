use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TaskMemory {
    pub task_summary: String,
    pub approach: String,
    pub final_answer: String,
    pub is_correct: bool,
    pub error_analysis: Option<String>,
}

impl TaskMemory {
    pub fn to_embedding_text(&self) -> String {
        format!("Task: {}", self.task_summary)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "UPPERCASE")]
pub enum MemoryDecision {
    Add,
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DuplicateCheckResult {
    pub decision: MemoryDecision,
    /// Explanation for the decision
    pub reason: String,
}

impl DuplicateCheckResult {
    pub fn is_duplicate(&self) -> bool {
        self.decision == MemoryDecision::Skip
    }
}

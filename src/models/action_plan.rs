use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ActionPlan {
    pub goal: String,
    pub steps: Vec<ActionStep>,
    pub difficulty: Difficulty,
    pub estimated_minutes: u32
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ActionStep {
    pub index: u8,
    pub description: String,
    pub tool_hint: Option<String>
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard
}
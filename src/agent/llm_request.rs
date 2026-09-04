use crate::agent::{ContentItem, ExecutionContext};

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
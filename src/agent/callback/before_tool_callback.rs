use async_openai::types::chat::ChatCompletionMessageToolCall;

use crate::agent::context::ExecutionContext;

pub mod approval_callback;

#[async_trait::async_trait]
pub trait BeforeToolCallback: Send + Sync {
    async fn call(
        &self,
        context: &ExecutionContext,
        tool_call: &ChatCompletionMessageToolCall,
    ) -> Option<String>;
}

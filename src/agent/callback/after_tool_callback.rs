use async_openai::types::chat::ChatCompletionMessageToolCall;

use crate::agent::context::ExecutionContext;

pub mod search_compressor;

#[async_trait::async_trait]
pub trait AfterToolCallback: Send + Sync {
    async fn call(
        &self,
        context: &ExecutionContext,
        tool_call: &ChatCompletionMessageToolCall,
        tool_result: &str,
    ) -> Option<String>;
}

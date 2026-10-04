use async_openai::types::chat::ChatCompletionMessageToolCall;

use crate::{
    agent::{callback::after_tool_callback::AfterToolCallback, context::ExecutionContext},
    search::{chunking::fixed_length_chunking, vector_search::vector_search},
    tool::web_search::WebSearchArgs,
};

pub struct SearchCompressor;

#[async_trait::async_trait]
impl AfterToolCallback for SearchCompressor {
    async fn call(
        &self,
        _context: &ExecutionContext,
        tool_call: &ChatCompletionMessageToolCall,
        tool_result: &str,
    ) -> Option<String> {
        if tool_call.function.name != "web_search" {
            return None;
        }

        if tool_result.len() < 2000 {
            return None;
        }

        let args: WebSearchArgs = serde_json::from_str(&tool_call.function.arguments).ok()?;

        let chunks = fixed_length_chunking(tool_result, 500, 50);
        let results = vector_search(&args.query, &chunks, 3).await.ok()?;
        let compressed = results
            .iter()
            .map(|sh| sh.text.clone())
            .collect::<Vec<_>>()
            .join("\n\n");

        Some(compressed)
    }
}

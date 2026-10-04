use std::collections::HashMap;
use std::sync::LazyLock;

use crate::{
    agent::{
        callback::before_llm_callback::context_optimizer::{ContextOptimizer, count_token},
        event::{ContentItem, ToolCall, ToolResult},
    },
    llm::llm_request::LlmRequest,
    session::model::Session,
};

static TOOLCALL_COMPACTION_RULES: LazyLock<HashMap<&'static str, &'static str>> =
    LazyLock::new(|| HashMap::from([("create_file", "[Content saved to file]")]));

static TOOLRESULT_COMPACTION_RULES: LazyLock<HashMap<&'static str, &'static str>> =
    LazyLock::new(|| {
        HashMap::from([
            (
                "read_file",
                "File content from {file_path}. Re-read if needed.",
            ),
            (
                "search_web",
                "Search results processed. Query: {query}. Re-search if needed.",
            ),
            (
                "tavily_search",
                "Search results processed. Query: {query}. Re-search if needed.",
            ),
        ])
    });

impl ContextOptimizer {
    pub fn compact(&self, _session: &Session, llm_request: &mut LlmRequest) {
        let token_count_before = count_token(&self.model, llm_request);
        if token_count_before < self.token_threshold {
            return;
        }

        tracing::info!(
            "ContextOptimizer: {} tokens >= threshold {}, compacting",
            token_count_before,
            self.token_threshold
        );

        let mut compacted = Vec::new();
        let mut tool_calls_args = HashMap::new();

        llm_request.contents.iter().for_each(|ci| match ci {
            ContentItem::ToolCall(tc) => {
                let tool_call_id = &tc.tool_call_id;
                tool_calls_args.insert(tool_call_id.to_owned(), tc.arguments.clone());
                if let Some(rule) = TOOLCALL_COMPACTION_RULES.get(tc.name.as_str()) {
                    let mut compressed_args = tc.arguments.clone();
                    if let Some(args) = compressed_args.as_object_mut() {
                        args.insert(
                            "content".to_owned(),
                            serde_json::Value::String(rule.to_string()),
                        );
                    }
                    compacted.push(ContentItem::ToolCall(ToolCall::new(
                        tool_call_id.clone(),
                        tc.name.clone(),
                        compressed_args,
                    )));
                } else {
                    compacted.push(ContentItem::ToolCall(tc.clone()));
                }
            }
            ContentItem::ToolResult(tr) => {
                if let Some(rule) = TOOLRESULT_COMPACTION_RULES.get(tr.name.as_str()) {
                    let tool_call_id = &tr.tool_call_id;
                    if let Some(tool_call_args) = tool_calls_args.get(tool_call_id) {
                        let compressed_content = rule
                            .replace(
                                "{file_path}",
                                tool_call_args
                                    .get("file_path")
                                    .or_else(|| tool_call_args.get("path"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("unknown"),
                            )
                            .replace(
                                "{query}",
                                tool_call_args
                                    .get("query")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("unknown"),
                            );

                        let tool_result = ContentItem::ToolResult(ToolResult::new(
                            tool_call_id.clone(),
                            tr.name.clone(),
                            tr.status.clone(),
                            compressed_content,
                        ));
                        compacted.push(tool_result);
                        return;
                    }
                }
                compacted.push(ContentItem::ToolResult(tr.clone()));
            }
            ContentItem::Message(msg) => {
                compacted.push(ContentItem::Message(msg.clone()));
            }
        });
        llm_request.contents = compacted;

        let token_count_after = count_token(&self.model, llm_request);
        tracing::info!(
            "ContextOptimizer: {} -> {} tokens",
            token_count_before,
            token_count_after
        );
    }
}
